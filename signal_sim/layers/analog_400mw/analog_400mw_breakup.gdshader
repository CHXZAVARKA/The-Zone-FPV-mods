shader_type canvas_item;

// Analog 400mw by Nito - the breakup: a signal_sim layer drawn on top of the
// game's own analog modes, after its link pass, over the picture and the OSD.
//
// The picture before it is the game's: its camera (analog_v2.gdshader) and its
// transmitter, link and receiver (analog_link.gdshader: bandwidth,
// cross-colour, FM noise), given only part of the signal loss (game_noise in
// signal_sim's settings) so its colour speckle, tears and roll stay mild.
// This pass does the rest, from 400 mW whoop DVR footage:
//   colour         the receiver carries colour narrower than brightness, so
//                  colour noise comes out as soft smears, not speckle; far out
//                  the colour fades and flickers
//   white noise    none on a strong signal; then white specks and short
//                  streaks, more and more with range and walls; at the end
//                  white snow over a grey picture
//   breakup bands  a stretch of lines where the colour locks wrong for a few
//                  frames - one hue per band, saturation thrown up - starting
//                  part way along some lines, patchy; on a weak signal with
//                  white sparkle through it; more often and taller with
//                  range and walls
//   colour hits    close in too: for a frame, below one line the hue is
//                  thrown off and swings back, darker and lighter parts each
//                  their own way, in bursts
// Timing follows the video: 59.94 fields a second, 240 lines a field at
// resolution 1 (the game's analog_common).

#include "res://ingame/analog_common.gdshaderinc"

uniform sampler2D screen_tex : hint_screen_texture, filter_linear, repeat_disable;
uniform float signal_quality = 100.0;   // 0-100, set every frame

uniform float strength = 1.0;           // all of the breakup below: 0 = none, 2 = twice as much
uniform float chroma_smooth = 1.0;      // colour carried softer than brightness (smears, not speckle)
uniform float white_noise = 1.0;        // white specks and streaks with range
uniform float snow = 1.0;               // white snow at the end of range
uniform float colour_fade = 1.0;        // colour fading and flickering far out
uniform float band_rate = 1.0;          // how often breakup bands come
uniform float band_period = 0.35;       // seconds a band slot lasts
uniform float band_height = 40.0;       // tallest band, in field lines (240 a field)
uniform float band_colour = 1.0;        // how strongly a band turns its hue
uniform float band_sparkle = 1.0;       // white sparkle inside a band
uniform float tear = 0.2;               // lines in a band pushed sideways
uniform float hit_rate = 0.5;           // how often colour hits come
uniform float hit_strength = 1.0;       // how far a hit throws the colour

float h21(vec2 p) {
	return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 rgb_to_yuv(vec3 c) {
	float y = dot(c, LUMA);
	return vec3(y, 0.492 * (c.b - y), 0.877 * (c.r - y));
}

// A band channel at time t: x = on (fading at its end), y = centre (0-1 of
// the height), z = height (0-1), w = its seed. Three channels each roll once
// a band_period for an arrival.
vec4 band_event(int bi, float t, float bad, float lines) {
	float et = t / max(band_period, 0.02) + float(bi) * 0.37;
	float slot = floor(et);
	float st = fract(et);
	float sd = mod(slot, 2048.0) * 3.137 + float(bi) * 17.0;
	// about one every 4-5 s on a strong link, a few a second far out
	float ep = strength * band_rate * (0.0149 + pow(bad, 1.5) * 0.36);
	float on = 1.0 - step(ep, h21(vec2(sd, 29.0)));
	float dur = 0.10 + 0.45 * h21(vec2(sd, 41.0));
	on *= 1.0 - smoothstep(dur * 0.7, dur, st);
	float cen = h21(vec2(sd, 53.0)) + st * 0.02;
	float hgt = (2.0 + max(band_height, 2.0) * (0.25 + 0.75 * bad) * pow(h21(vec2(sd, 71.0)), 1.5)) / lines;
	return vec4(on, cen, hgt, sd);
}

// A short white dash in this cell of the line, or 0: bright where it starts,
// its tail fading to the right, as FM clicks show on the DVR. dens = the
// chance this cell has one.
float white_dash(float x_abs, float cell_w, float dens, uint seed) {
	float cell = floor(x_abs / cell_w);
	uint cs = pcg(uint(int(cell)) ^ seed);
	if (rnd(1, cs) >= dens) {
		return 0.0;
	}
	float len = 1.0 + (cell_w - 2.0) * pow(rnd(2, cs), 2.0);
	float x0 = cell * cell_w + rnd(3, cs) * max(cell_w - len - 1.0, 0.3);
	float d = x_abs - x0;
	return (0.55 + 0.45 * rnd(4, cs)) * smoothstep(-0.6, 0.2, d) * (1.0 - smoothstep(len * 0.3, len + 0.8, d));
}

void fragment() {
	vec2 suv = SCREEN_UV;
	vec2 tex_size = 1.0 / SCREEN_PIXEL_SIZE;
	float aspect = tex_size.x / tex_size.y;
	int field = int(floor(TIME * FIELD_RATE));
	float lines = LINES * resolution;
	float line = floor(suv.y * lines + 0.5 * float(field & 1));
	float apx_w = ANALOG_PX * resolution * aspect;      // analog pixels across the screen
	float x_abs = suv.x * apx_w;
	float fld = float(field % 65536);
	uint field_seed = pcg(uint(field) + 77u);
	uint line_seed = pcg(uint(int(line)) ^ field_seed);  // new on every line of every field
	float q = clamp(signal_quality / 100.0, 0.0, 1.0);
	float bad = clamp((0.98 - q) / 0.98, 0.0, 1.0);    // as the game's link pass has it

	// ---- breakup bands ----
	float bmask = 0.0;
	float brot = 0.0;
	float bgain = 1.0;
	vec2 badd = vec2(0.0);
	float shift = 0.0;
	for (int bi = 0; bi < 3; bi++) {
		vec4 be = band_event(bi, TIME, bad, lines);
		if (be.x <= 0.0) {
			continue;
		}
		float sd = be.w;
		// ragged ends: each line reaches its own way past the band's edge
		float half_h = be.z * 0.5;
		float rag = (h21(vec2(line, sd + 5.0)) - 0.5) * half_h * 0.9
			+ (vnoise(line / 3.0, uint(sd * 7.0)) - 0.5) * half_h * 0.8;
		float d = abs(suv.y - be.y) - half_h - rag;
		float inb = be.x * (1.0 - smoothstep(-1.5 / lines, 1.5 / lines, d));
		if (inb <= 0.0) {
			continue;
		}
		// not a bar: on some lines the hue goes wrong only part way along (a
		// click knocks the colour out of lock until the next line), each line
		// its own strength, and patchy across the width
		float onset = pow(h21(vec2(line, sd + 23.0)), 2.2) * 0.85 - 0.1;
		float along = smoothstep(onset, onset + 0.06, suv.x);
		float patch = mix(0.35, 1.0, smoothstep(0.2, 0.7, vnoise(x_abs / 70.0 + line / 9.0, uint(sd * 11.0) + 3u)));
		inb *= along * patch * (0.45 + 0.55 * h21(vec2(line, sd + 31.0)));
		bmask = max(bmask, inb);
		brot += inb * (h21(vec2(sd, 97.0)) - 0.5) * 2.0 * band_colour * 2.4;
		bgain += inb * band_colour * (0.3 + 1.0 * h21(vec2(sd, 113.0)));
		vec2 bdir = normalize(vec2(h21(vec2(sd, 131.0)) - 0.5, h21(vec2(sd, 149.0)) - 0.5) + 0.0001);
		badd += bdir * inb * band_colour * (0.03 + 0.16 * pow(h21(vec2(sd, 167.0)), 2.0)) * (0.4 + 0.6 * bad);
		// a few lines torn sideways, new every field
		shift += inb * tear * (h21(vec2(line, fld + sd)) - 0.5) * (2.0 + 10.0 * bad);
	}

	// ---- the picture, colour carried soft ----
	// 7 taps along the line, 1.5 analog pixels apart: brightness from the
	// middle one, colour from all, weighted by a gaussian chroma_smooth * 2.5
	// analog pixels wide
	vec2 p = suv - vec2(shift / apx_w, 0.0);
	float sig = max(chroma_smooth * 2.5, 0.05);
	float y = 0.0;
	vec2 c = vec2(0.0);
	float wsum = 0.0;
	for (int k = -3; k <= 3; k++) {
		float o = float(k) * 1.5;
		float w = exp(-o * o / (2.0 * sig * sig));
		vec3 yuv = rgb_to_yuv(texture(screen_tex, p + vec2(o / apx_w, 0.0)).rgb);
		if (k == 0) {
			y = yuv.x;
		}
		c += w * yuv.yz;
		wsum += w;
	}
	vec2 uv2 = c / wsum;

	// ---- colour fading far out ----
	// the receiver's colour killer: in and out, a few times a second
	float flick = vnoise(TIME * 5.0, 41u) - 0.5;
	uv2 *= 1.0 - 0.6 * clamp(strength, 0.0, 1.0) * colour_fade * smoothstep(0.6, 1.0, bad + 0.2 * flick);

	// ---- colour hits ----
	// 30 slots a second, in bursts: a busy second or two, then quieter
	float hphi = 0.0;
	float hgain = 1.0;
	float hdp = 0.0;
	float hz = 0.0;
	float hs0 = floor(TIME * 30.0);
	for (int j = 0; j < 2; j++) {
		float s = hs0 - float(j);
		float bt = s / 30.0 * 0.7;
		float bi2 = mod(floor(bt), 4096.0);
		float bf = fract(bt);
		bf = bf * bf * (3.0 - 2.0 * bf);
		float bn = mix(h21(vec2(bi2, 401.0)), h21(vec2(mod(bi2 + 1.0, 4096.0), 401.0)), bf);
		float hp = strength * hit_rate * (3.2 + 4.0 * bad) * mix(0.12, 3.0, bn * bn) / 30.0;
		float sd = mod(s, 4096.0) * 1.713 + mod(floor(s / 4096.0), 1024.0) * 0.3571;
		if (h21(vec2(sd, 403.0)) >= hp) {
			continue;
		}
		float hd = h21(vec2(sd, 411.0));
		float dur = 1.0 + step(0.8, hd);
		if (float(j) >= dur) {
			continue;
		}
		float y0 = 0.02 + 0.9 * h21(vec2(sd, 419.0));
		float dd = (line - floor(y0 * lines)) / lines;
		if (dd < 0.0) {
			continue;
		}
		float lr = 0.12 + 0.55 * pow(h21(vec2(sd, 431.0)), 0.8);
		float z = 1.0 - smoothstep(0.3 * lr, lr, dd);
		float sg = step(0.5, h21(vec2(sd, 437.0))) * 2.0 - 1.0;
		hphi += (0.8 + 2.3 * h21(vec2(sd, 433.0))) * sg * z * hit_strength;
		hdp += (step(0.5, h21(vec2(sd, 453.0))) * 2.0 - 1.0) * (3.0 + 9.0 * h21(vec2(sd, 455.0))) * z * hit_strength;
		hgain *= 1.0 + (0.4 + 1.6 * h21(vec2(sd, 449.0))) * hit_strength * z;
		hz = max(hz, z);
	}

	// colour: a hit turns the hue by brightness too (light and dark parts
	// each their own way), a band one hue across; faint colour stays faint
	float core = smoothstep(0.01, 0.04, length(uv2));
	float ang = (hphi + hdp * (y - 0.4)) * core + brot;
	float ca = cos(ang);
	float sa = sin(ang);
	uv2 = vec2(ca * uv2.x - sa * uv2.y, sa * uv2.x + ca * uv2.y);
	uv2 *= mix(1.0, hgain, hz * core) * mix(1.0, bgain, max(core, bad));
	uv2 += badd;
	// far out a band lifts the blacks a little: the noise in it is light
	y = mix(y, 0.12 + 0.88 * y, bmask * 0.8 * bad);

	// ---- white noise ----
	// white specks and short streaks one line tall, boiling field to field:
	// none on a strong signal (quality above about 92), then more and more
	// with range; inside a band too, once the signal is weak
	float wn = max(bad - 0.06, 0.0) / 0.94;
	float dens = strength * (white_noise * 0.5 * pow(wn, 2.6) + band_sparkle * bmask * 0.45 * smoothstep(0.05, 0.6, bad));
	if (dens > 0.0) {
		float wd = white_dash(x_abs, 8.0, dens, line_seed);
		y = mix(y, 0.92 + 0.08 * rnd(9, line_seed), wd);
		uv2 *= 1.0 - 0.9 * wd;
	}

	// ---- snow ----
	// the picture sinks into grey with white streaks through it, nearly
	// without colour; it comes through better on some stretches of lines
	float sn = clamp(strength, 0.0, 1.0) * snow * smoothstep(0.66, 1.06, bad);
	if (sn > 0.0) {
		sn = clamp(sn + 0.5 * sn * (1.0 - sn) * (vnoise(line / 12.0, pcg(uint(field / 2) + 5u)) * 2.0 - 1.0), 0.0, 1.0);
		float g1 = gnoise(x_abs / 2.5, pcg(line_seed + 1u));
		float g2 = gnoise(x_abs / 14.0, pcg(line_seed + 2u));
		float wht = smoothstep(0.1, 0.5, g1 + 0.8 * g2);
		float y_snow = mix(0.28 + 0.25 * g1, 0.95, wht);
		y = mix(y, y_snow, sn);
		uv2 *= 1.0 - 0.9 * sn;
	}

	COLOR = vec4(clamp(yuv_to_rgb(y, uv2.x, uv2.y), 0.0, 1.0), 1.0);
}

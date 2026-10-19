"""Light pages by location.

The 2026 beta draws a baked map as one node per light page, and Godot culls
whole nodes: behind the camera (frustum) or behind the map's occluders
(occluder.bin). A page packed by size alone holds charts from all over the
map - gm_br_pitfalls' 32 pages each spanned nearly the whole 536 x 652 m map -
so nothing could be culled: every frame drew all 7.5 M triangles in ~5,000
draw calls, wherever the drone was.

So charts are first grouped by where they are: the map is split in two along
its longest side - in proportion to the pages each side needs - again and
again, until a group
fits one page (with room for the shelf packer's waste) and has at most 256
materials (Godot's limit on a mesh's surfaces - a page over it would have to
be several nodes, each loading its own copy of the page's light). Each group
is then shelf-packed on its own, so each page - each node - is one compact
part of the map.
"""
import numpy as np

PAD = 2                 # texels round each chart (bake_glb's PAD)
FILL = 0.88             # how full a group may make a page (the skyline packer fits 90 %)
MAX_SURFACES = 256      # Godot's RS::MAX_MESH_SURFACES


def regions(cen, area, mats, page_px, fill=FILL, max_mats=MAX_SURFACES):
    """cen (n, 3) chart centres, area (n,) texels each with its padding, mats
    (n,) material ids -> [index arrays], each a group for one page"""
    cap = fill * page_px * page_px
    out, todo = [], [np.arange(len(cen))]
    while todo:
        idx = todo.pop()
        if len(idx) <= 1 or (area[idx].sum() <= cap and len(np.unique(mats[idx])) <= max_mats):
            out.append(idx)
            continue
        c = cen[idx]
        ax = int(np.argmax(c.max(0) - c.min(0)))
        order = idx[np.argsort(c[:, ax], kind='stable')]
        cum = np.cumsum(area[order])
        # in proportion to the pages each side will need, not always in half:
        # halving a group that needs 5 pages gives 2.5 + 2.5 -> 3 + 3, and a
        # map just over 4 pages came out as 8 at 35 % (de_dust2)
        n = max(2, int(np.ceil(cum[-1] / cap)))
        k = int(np.searchsorted(cum, cum[-1] * (n // 2) / n))
        k = min(max(k, 1), len(order) - 1)
        todo += [order[:k], order[k:]]
    return out


def _skyline_page(items, W, H):
    """Skyline bottom-left packing of (index, w, h) onto one W x H page, tallest
    first: -> {index: (x, y)} for those that fit, and the rest"""
    sky = [[0, 0, W]]                 # segments: x, y (height reached), width
    placed, rest = {}, []
    for i, w, h in items:
        best = None
        for si in range(len(sky)):
            x = sky[si][0]
            if x + w > W:
                break
            # the height of the skyline under [x, x + w)
            y, left, sj = 0, w, si
            while left > 0:
                y = max(y, sky[sj][1])
                left -= sky[sj][2]
                sj += 1
            if y + h <= H and (best is None or y < best[1] or (y == best[1] and x < best[0])):
                best = (x, y, si)
        if best is None:
            rest.append((i, w, h))
            continue
        x, y, si = best
        placed[i] = (x, y)
        # the new segment, and what it covers
        new = [x, y + h, w]
        end = x + w
        sj = si
        while sj < len(sky) and sky[sj][0] < end:
            seg_end = sky[sj][0] + sky[sj][2]
            if seg_end > end:            # the tail sticks out: keep it
                sky[sj] = [end, sky[sj][1], seg_end - end]
                break
            sj += 1
        sky[si:sj] = [new]
        # merge neighbours at the same height
        k = max(si - 1, 0)
        while k < len(sky) - 1 and k <= si + 1:
            if sky[k][1] == sky[k + 1][1]:
                sky[k][2] += sky[k + 1][2]
                del sky[k + 1]
            else:
                k += 1
    return placed, rest


def shelf(tw, th, page_px, first_page=0):
    """Pack charts (texel sizes tw, th) onto pages from first_page - skyline,
    bottom-left, tallest first (bake_glb's shelf_pack only tries the last
    shelf, and wastes about half of a page of mixed sizes): -> page, x, y per
    chart (x, y inside the padding), pages used"""
    n = len(tw)
    page = np.zeros(n, np.int64); x = np.zeros(n, np.int64); y = np.zeros(n, np.int64)
    items = sorted(((i, int(tw[i]) + 2 * PAD, int(th[i]) + 2 * PAD) for i in range(n)),
                   key=lambda t: (-t[2], -t[1]))
    p = first_page
    while items:
        placed, items = _skyline_page(items, page_px, page_px)
        if not placed:
            raise ValueError('a chart bigger than a page')
        for i, (px, py) in placed.items():
            page[i], x[i], y[i] = p, px + PAD, py + PAD
        p += 1
    return page, x, y, sorted(set(int(q) for q in page)) if n else []


def layout(cen, tw, th, mats, page_px):
    """Charts onto pages by location -> page, x, y per chart, number of pages"""
    tw = np.asarray(tw, np.int64); th = np.asarray(th, np.int64)
    area = ((tw + 2 * PAD) * (th + 2 * PAD)).astype(np.float64)
    page = np.zeros(len(tw), np.int64); x = np.zeros_like(page); y = np.zeros_like(page)
    nxt = 0
    groups = regions(np.asarray(cen, np.float64), area, np.asarray(mats), page_px)
    # pack the groups in a spatial order, so page numbers run across the map
    groups.sort(key=lambda g: tuple(np.round(np.asarray(cen)[g].mean(0) / 50.0)))
    for g in groups:
        pg, gx, gy, used = shelf(tw[g], th[g], page_px, nxt)
        page[g], x[g], y[g] = pg, gx, gy
        nxt = max(used) + 1 if used else nxt
    return page, x, y, nxt

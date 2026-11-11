#!/usr/bin/env python3
"""Browse TIFF stacks locally without loading entire stacks into memory."""
import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser

import numpy as np
from PIL import Image, ImageDraw

PAGE = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>TIFF viewer</title><style>
body{font:14px system-ui;background:#181a1e;color:#eee;margin:0} header{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px;background:#292c32} select{max-width:50vw} button,input,select{font:inherit} main{display:flex;justify-content:center;align-items:center;height:calc(100vh - 125px);overflow:auto;background:#090a0c} img{max-width:100%;max-height:100%;image-rendering:pixelated} footer{padding:10px;display:flex;gap:10px;align-items:center} #frame{flex:1} #status{color:#bbb} label{white-space:nowrap} input[type=number]{width:75px}
</style></head><body><header><select id="file"></select><button id="play">▶ Play</button><label>FPS <input id="fps" type="number" min="0.1" max="60" step="0.5" value="4"></label><label>Black <input id="low" type="number" value="0"></label><label>White <input id="high" type="number" value="65535"></label><button id="auto">Auto contrast</button><label>Min. net gradient <input id="gradient" type="number" min="0" step="100" value="3000"></label><button id="prepare">Prepare correction</button><select id="mode"><option value="raw">Raw</option><option value="localisations">Localisations (raw)</option><option value="registered">Registered</option><option value="corrected">Drift corrected</option><option value="overlay">Corrected overlay</option></select><span id="progress"></span><span id="status">Opening…</span></header><main><img id="image" draggable="false"></main><footer><button id="prev">◀</button><input id="frame" type="range" min="0" value="0"><button id="next">▶</button><span id="position"></span></footer><script>
const $=id=>document.getElementById(id); let count=1, current=0, running=false, timer=null, busy=false, pending=false, generation=0;
async function openFile(){running=false; $('play').textContent='▶ Play'; clearTimeout(timer); generation++; current=0; $('status').textContent='Opening…'; try {let r=await fetch('/info?file='+$('file').value); if(!r.ok) throw Error(await r.text()); let d=await r.json(); count=d.frames; $('frame').max=count-1; $('low').value=d.low; $('high').value=d.high; $('fps').value=d.fps||4; $('status').textContent=`${d.width} × ${d.height}, ${d.mode}; ${count} frames${d.expected && d.expected!==count ? ' (metadata says '+d.expected+')' : ''}`; show();}catch(e){$('status').textContent=e.message}}
function show(){ $('frame').value=current; $('position').textContent=`${current+1} / ${count}`; pending=true; load() }
async function load(){if(busy||!pending)return; pending=false; busy=true; const g=generation; const params=new URLSearchParams({file:$('file').value,frame:current,low:$('low').value,high:$('high').value,mode:$('mode').value,gradient:$('gradient').value}); try {let r=await fetch('/frame?'+params); if(!r.ok)throw Error(await r.text()); let blob=await r.blob(); if(g===generation&&!pending){let url=URL.createObjectURL(blob), old=$('image').dataset.url; $('image').src=url; $('image').dataset.url=url; if(old)URL.revokeObjectURL(old); if(params.get('mode')==='localisations') $('progress').textContent='Localisations: left '+r.headers.get('X-Left-Count')+', right '+r.headers.get('X-Right-Count')}}catch(e){if(g===generation)$('status').textContent=e.message} finally{busy=false;if(pending)load();else if(running)schedule()}}
function schedule(){clearTimeout(timer); timer=setTimeout(()=>{if(running){current=(current+1)%count;show()}},1000/Math.max(.1,Number($('fps').value)||4))}
$('mode').onchange=show; $('gradient').onchange=()=>{if($('mode').value==='localisations')show(); $('progress').textContent='Gradient changed; prepare correction again to use it'}; $('file').onchange=()=>{ $('mode').value='raw'; $('progress').textContent=''; openFile() };
$('prepare').onclick=async()=>{let file=$('file').value; $('prepare').disabled=true; $('progress').textContent='Starting…'; try {let r=await fetch('/prepare?'+new URLSearchParams({file,gradient:$('gradient').value}),{method:'POST'}); let d=await r.json(); if(!r.ok)throw Error(d.error); while(true){await new Promise(resolve=>setTimeout(resolve,700)); let res=await fetch('/prepare?file='+file); let state=await res.json(); if(file===$('file').value) $('progress').textContent=state.message; if(state.state==='done'){if(file===$('file').value){$('mode').value='corrected';show()} break} if(state.state==='error')throw Error(state.message);}}catch(e){if(file===$('file').value)$('progress').textContent=e.message}finally{$('prepare').disabled=false}};
 $('frame').oninput=e=>{current=Number(e.target.value);show()}; $('prev').onclick=()=>{current=Math.max(0,current-1);show()}; $('next').onclick=()=>{current=Math.min(count-1,current+1);show()}; $('play').onclick=()=>{running=!running;$('play').textContent=running?'❚❚ Pause':'▶ Play';if(running&&!busy)schedule();else if(!running)clearTimeout(timer)}; for(const id of ['low','high']) $(id).onchange=show;
$('auto').onclick=async()=>{let r=await fetch('/contrast?file='+$('file').value+'&frame='+current);if(r.ok){let d=await r.json();$('low').value=d.low;$('high').value=d.high;show()}};
window.onkeydown=e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft')$('prev').click();if(e.key==='ArrowRight')$('next').click();if(e.key===' '){e.preventDefault();$('play').click()}};
fetch('/files').then(r=>r.json()).then(files=>{files.forEach((name,i)=>{let o=document.createElement('option');o.value=i;o.textContent=name;$('file').appendChild(o)});openFile()});
</script></body></html>'''

def preprocess_frame(frame, homography, drift_xy=(0.0, 0.0)):
    """Return left and registered right frames, drift-corrected in left pixels.

    Does not mutate the source. Homography maps right-half to left-half
    coordinates; drift_xy is measured in left coordinates for this frame.
    """
    import cv2
    from scipy.ndimage import shift

    if frame.ndim != 2 or frame.shape[1] % 2:
        raise ValueError("Expected a grayscale frame with two equal-width halves")
    height, width = frame.shape
    midpoint = width // 2
    left = frame[:, :midpoint].astype(np.float32)
    right = frame[:, midpoint:].astype(np.float32)
    right = cv2.warpPerspective(right, homography, (midpoint, height),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    dx, dy = drift_xy
    if dx or dy:
        left = shift(left, (-dy, -dx), mode='constant', cval=0.0)
        right = shift(right, (-dy, -dx), mode='constant', cval=0.0)
    return left, right


def localise_frames(frames, min_gradient, start_frame=0):
    """Use the same Picasso detection and fitting settings throughout the viewer."""
    from picasso.localize import localize_frames

    if not np.isfinite(min_gradient) or min_gradient < 0:
        raise ValueError('Min. net gradient must be a non-negative finite number')
    return localize_frames(
        frames, info=None,
        params={'Box Size': 7, 'Min. Net Gradient': min_gradient},
        camera_info={'Baseline': 0, 'Sensitivity': 1.0, 'Gain': 1, 'Pixelsize': 117},
        start_frame=int(start_frame), fitting_method='gausslq', threaded=False,
    )


def prepare_correction(path, progress, frame_stride=10, segment_frames=300, batch_size=16,
                       min_gradient=3000):
    """Estimate the notebook's frame-0 homography and sampled Picasso RCC drift."""
    import cv2
    import pandas as pd
    from picasso.postprocess import undrift

    if frame_stride < 1 or batch_size < 1:
        raise ValueError('frame_stride and batch_size must be positive')

    def localise(image, frame_index):
        return localise_frames(image[np.newaxis], min_gradient, frame_index)

    with Image.open(path) as movie:
        n_frames = movie.n_frames
        width, height = movie.size
        if width % 2:
            raise ValueError('Expected two equal-width channels')
        midpoint = width // 2
        first = np.array(movie)
        progress('Fitting calibration spots in frame 0…')
        reference = localise(first[:, :midpoint], 0)[['x', 'y']].to_numpy(dtype=np.float32)
        target = localise(first[:, midpoint:], 0)[['x', 'y']].to_numpy(dtype=np.float32)
        if not len(reference) or not len(target):
            raise ValueError('No calibration localisations in one channel')
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = sorted(matcher.match(reference, target), key=lambda m: m.distance)
        if len(matches) < 4:
            raise ValueError(f'Only {len(matches)} matched calibration spots; at least four needed')
        ref = np.float32([reference[m.queryIdx] for m in matches])
        tgt = np.float32([target[m.trainIdx] for m in matches])
        H, mask = cv2.findHomography(tgt, ref, cv2.RANSAC)
        if H is None or mask is None or not np.isfinite(H).all():
            raise ValueError('Unable to estimate a valid homography')
        sample = np.arange(0, n_frames, frame_stride)
        batches = []
        for start in range(0, len(sample), batch_size):
            frame_numbers = sample[start:start + batch_size]
            images = []
            for frame_number in frame_numbers:
                movie.seek(int(frame_number))
                images.append(np.asarray(movie)[:, :midpoint].copy())
            locs = localise_frames(np.stack(images), min_gradient)
            if not locs.empty:
                # Picasso reports indices within this stack. The input frames
                # are sampled, so an offset alone cannot recover TIFF indices.
                locs['frame'] = frame_numbers[locs['frame'].to_numpy(dtype=int)]
                batches.append(locs)
            progress(f'Localising sampled frames: {start + len(frame_numbers)}/{len(sample)}')
    if not batches:
        raise ValueError('No localisations found for drift estimation')
    points = pd.concat(batches, ignore_index=True)
    n_segments = int(np.round(n_frames / segment_frames))
    if n_segments < 4:
        raise ValueError('RCC needs at least four segments; this file is too short for the current settings')
    bounds = np.linspace(0, n_frames - 1, n_segments + 1, dtype=np.uint32)
    counts = [((points['frame'] >= bounds[i]) & (points['frame'] < bounds[i+1])).sum()
              for i in range(n_segments)]
    if min(counts) == 0:
        raise ValueError('An RCC segment has no localisations; sample more frames or adjust detection settings')
    progress('Estimating RCC drift…')
    info = [{'Frames': n_frames, 'Height': height, 'Width': midpoint,
             'Pixelsize': 117}, {}]
    drift, _ = undrift(points, info, segmentation=segment_frames, display=False)
    values = drift[['x', 'y']].to_numpy(dtype=float)
    if values.shape != (n_frames, 2) or not np.isfinite(values).all():
        raise ValueError('Picasso returned invalid drift values')
    values -= values[0].copy()
    return {'homography': H, 'drift': values, 'matches': len(matches),
            'inliers': int(mask.sum()), 'counts': counts}


class Stack:
    def __init__(self, paths):
        self.paths = paths
        self.lock = threading.Lock()
        self.index = None
        self.image = None
        self.cache = OrderedDict()

    def select(self, index):
        if not 0 <= index < len(self.paths):
            raise ValueError('Invalid file')
        if index != self.index:
            if self.image:
                self.image.close()
            self.image = Image.open(self.paths[index])
            self.index = index
            self.cache.clear()
        return self.image

    def frame(self, index, number):
        im = self.select(index)
        if not 0 <= number < im.n_frames: #type: ignore
            raise ValueError('Invalid frame')
        if number not in self.cache:
            im.seek(number) # type: ignore
            self.cache[number] = np.array(im)
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(number)
        return self.cache[number]


def contrast(frame):
    # Sampling avoids scanning every pixel just to establish display limits.
    sample = frame[::4, ::4]
    low, high = np.percentile(sample, (1, 99.8))
    return int(low), max(int(low) + 1, int(high))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', nargs='?', type=Path, help='TIFF file or directory (default: both example folders)')
    parser.add_argument('--port', type=int, default=0, help='Local port (default: choose automatically)')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if args.path is None:
        root = Path(__file__).parent
        paths = [p for folder in ('low_density_examples', 'high_density_examples')
                 for p in sorted((root / folder).glob('*'))
                 if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')]
    else:
        paths = sorted(p for p in args.path.iterdir() if p.is_file() and p.suffix.lower() in ('.tif', '.tiff')) if args.path.is_dir() else [args.path]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('No TIFF files found')
    stack = Stack(paths)
    jobs = {}
    jobs_lock = threading.Lock()

    def run_preparation(index, min_gradient):
        def progress(message):
            with jobs_lock:
                jobs[index]['message'] = message
        try:
            result = prepare_correction(paths[index], progress, min_gradient=min_gradient)
            with jobs_lock:
                jobs[index].update(state='done', result=result,
                    message=f"Ready (gradient {min_gradient:g}): {result['inliers']}/{result['matches']} calibration matches are inliers")
        except Exception as exc:
            with jobs_lock:
                jobs[index].update(state='error', message=f'Correction failed: {exc}')


    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            url = urlsplit(self.path)
            try:
                query = parse_qs(url.query)
                index = int(query['file'][0])
                gradient = float(query.get('gradient', ['3000'])[0])
                if not np.isfinite(gradient) or gradient < 0:
                    raise ValueError('Min. net gradient must be a non-negative finite number')
                if url.path != '/prepare' or not 0 <= index < len(paths):
                    raise ValueError('Invalid file or action')
                with jobs_lock:
                    if any(job['state'] == 'running' for job in jobs.values()):
                        raise ValueError('A correction is already being prepared')
                    jobs[index] = {'state': 'running', 'message': 'Starting…'}
                threading.Thread(target=run_preparation, args=(index, gradient), daemon=True).start()
                self.json({'state': 'running'})
            except (KeyError, ValueError) as exc:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(exc)}).encode())

        def do_GET(self):
            try:
                url = urlsplit(self.path)
                q = parse_qs(url.query)
                if url.path == '/':
                    return self.respond(PAGE.encode(), 'text/html; charset=utf-8')
                if url.path == '/files':
                    return self.json([f'{p.parent.name}/{p.name}' for p in paths])
                index = int(q['file'][0])
                if url.path == '/prepare':
                    with jobs_lock:
                        job = jobs.get(index, {'state': 'idle', 'message': 'Not prepared'})
                        return self.json({'state': job['state'], 'message': job['message']})
                with stack.lock:
                    im = stack.select(index)
                    if url.path == '/info':
                        metadata = {}
                        try:
                            metadata = json.loads(im.tag_v2.get(270, '{}').rstrip('\x00')) #type: ignore
                        except (ValueError, TypeError):
                            pass
                        n = im.n_frames #type: ignore
                        low, high = contrast(stack.frame(index, 0))
                        return self.json(dict(width=im.width, height=im.height, mode=im.mode, frames=n, low=low, high=high, fps=metadata.get('FramesPerSecond'), expected=metadata.get('Frames'))) #type: ignore
                    number = int(q['frame'][0])
                    arr = stack.frame(index, number)
                    if url.path == '/contrast':
                        low, high = contrast(arr)
                        return self.json(dict(low=low, high=high))
                    if url.path == '/frame':
                        mode = q.get('mode', ['raw'])[0]
                        if mode not in ('raw', 'localisations', 'registered', 'corrected', 'overlay'):
                            raise ValueError('Invalid display mode')
                        if mode not in ('raw', 'localisations'):
                            with jobs_lock:
                                job = jobs.get(index, {})
                                result = job.get('result')
                            if result is None:
                                raise ValueError('Select Prepare correction first and wait for it to finish')
                            drift = result['drift'][number] if mode != 'registered' else (0.0, 0.0)
                            left, right = preprocess_frame(arr, result['homography'], drift)
                        low, high = float(q['low'][0]), float(q['high'][0])
                        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                            raise ValueError('White must be greater than black')
                        def scale(image):
                            return np.clip((image.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
                        counts = None
                        if mode in ('raw', 'localisations'):
                            pixels = scale(arr)
                            if mode == 'localisations':
                                if arr.ndim != 2 or arr.shape[1] % 2:
                                    raise ValueError('Expected a grayscale frame with two equal-width halves')
                                gradient = float(q.get('gradient', ['3000'])[0])
                                midpoint = arr.shape[1] // 2
                                l = localise_frames(arr[np.newaxis, :, :midpoint], gradient)
                                r = localise_frames(arr[np.newaxis, :, midpoint:], gradient)
                                result_image = Image.fromarray(pixels).convert('RGB')
                                draw = ImageDraw.Draw(result_image)
                                for locs, offset, colour in ((l, 0, 'lime'), (r, midpoint, 'red')):
                                    for x, y in locs[['x', 'y']].to_numpy():
                                        if np.isfinite(x) and np.isfinite(y):
                                            x += offset
                                            draw.ellipse((x-4, y-4, x+4, y+4), outline=colour, width=1)
                                pixels = np.asarray(result_image)
                                counts = (len(l), len(r))
                        elif mode == 'overlay':
                            l, r = scale(left), scale(right)
                            pixels = np.stack((r, l, r), axis=-1)
                        else:
                            pixels = np.concatenate((scale(left), scale(right)), axis=1)
                        out = BytesIO()
                        Image.fromarray(pixels).save(out, format='PNG', compress_level=1)
                        headers = {} if counts is None else {'X-Left-Count': str(counts[0]), 'X-Right-Count': str(counts[1])}
                        return self.respond(out.getvalue(), 'image/png', headers)
                self.send_error(404)
            except (KeyError, ValueError, OSError, IndexError) as exc:
                self.send_error(400, str(exc))

        def json(self, obj):
            self.respond(json.dumps(obj).encode(), 'application/json')

        def respond(self, data, kind, headers=None):
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}/'
    print(f'Viewer: {url}\nPress Ctrl+C to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()

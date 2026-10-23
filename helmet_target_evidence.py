"""Target-bound helmet review images and an independent person verification gate."""
import base64
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法读取目标证据图')
    return image


def save_image(path, image):
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError('目标证据图编码失败')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data.tobytes())
    return Path(path)


def crop_target(image, box):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError('目标框无效')
    # Exact box: surrounding people cannot supply missing human/head evidence.
    return image[y1:y2, x1:x2].copy()


def build_target_evidence(original_path, box, keyframes, root, target_id):
    root = Path(root)
    original = read_image(original_path)
    crop = save_image(root/(target_id+'_tight.jpg'), crop_target(original, box))
    marked = original.copy()
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(marked, (x1,y1), (x2,y2), (0,0,255), 3)
    cv2.putText(marked, target_id, (x1,max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,255), 2)
    context = save_image(root/(target_id+'_context.jpg'), marked)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    mask[max(0,y1):min(gray.shape[0],y2),max(0,x1):min(gray.shape[1],x2)] = 255
    points = cv2.goodFeaturesToTrack(gray, 80, .02, 4, mask=mask)
    frames, diagnostics = [], []
    for i, frame in enumerate(keyframes, 1):
        item = dict(frame)
        image = read_image(frame['path'])
        reason = 'insufficient_track_points'
        if points is not None and len(points) >= 6 and image.shape == original.shape:
            current = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray, current, points, None)
            if nxt is not None:
                back, bst, _ = cv2.calcOpticalFlowPyrLK(current, gray, nxt, None)
                if back is not None:
                    good = (st.ravel()==1)&(bst.ravel()==1)&(np.linalg.norm(back-points,axis=2).ravel()<1.5)
                    delta = (nxt-points).reshape(-1,2)[good]
                    if len(delta) >= 6:
                        shift = np.median(delta,axis=0)
                        coherent = np.linalg.norm(delta-shift,axis=1) < max(2., .05*min(x2-x1,y2-y1))
                        if coherent.mean() >= .7 and good.mean() >= .5:
                            moved = [x1+shift[0],y1+shift[1],x2+shift[0],y2+shift[1]]
                            patch = crop_target(image,moved)
                            item.update(path=save_image(root/(target_id+'_T'+str(i)+'.jpg'),patch), target_box=list(map(float,moved)))
                            frames.append(item)
                            reason = 'tracked'
        diagnostics.append(dict(frame=i, result=reason))
    return crop, context, frames, diagnostics


def target_contact_sheet(frames, path):
    tiles = []
    for index, frame in enumerate(frames, 1):
        image = read_image(frame['path'])
        tile = np.zeros((384,320,3),np.uint8)
        scale = min(320/image.shape[1],348/image.shape[0])
        image = cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))))
        x = (320-image.shape[1])//2
        tile[36:36+image.shape[0],x:x+image.shape[1]] = image
        cv2.putText(tile,'T%d %.2fs'%(index,frame['seconds']),(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
        tiles.append(tile)
    while len(tiles)<6:
        tiles.append(np.zeros((384,320,3),np.uint8))
    return save_image(path,cv2.vconcat([cv2.hconcat(tiles[:3]),cv2.hconcat(tiles[3:6])]))


def verify_person(crop, target_id, server_url, model, timeout):
    schema = dict(type='object', additionalProperties=False,
        required=['target_id','person','confidence','features','summary'], properties={
        'target_id':dict(type='string'),
        'person':dict(type='string',enum=['yes','no','unclear']),
        'confidence':dict(type='number',minimum=0,maximum=1),
        'features':dict(type='array',items=dict(type='string',enum=['head','torso','arm','leg']),maxItems=4),
        'summary':dict(type='string')})
    prompt = ('只判断这张目标裁剪图内是否存在真实人体，不判断安全帽。墙体、水渍、阴影、文字、机械和衣物状纹理不等于人。'
              '必须有相互连接、位置合理的解剖结构，不能根据检测框或纹理想象人体。模糊、遮挡或不能确定时回答unclear。'
              'features只列实际可辨的人体结构。没有人时回答no。target_id原样返回：'+target_id)
    image = read_image(crop)
    scale = min(4., 512/max(image.shape[:2]))
    if scale > 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise ValueError('人体核验图片编码失败')
    body = dict(model=model, temperature=0, max_tokens=400, chat_template_kwargs={'enable_thinking':False},
        response_format={'type':'json_schema','json_schema':{'name':'person_gate','strict':True,'schema':schema}},
        messages=[dict(role='user',content=[dict(type='text',text=prompt),dict(type='image_url',image_url={
            'url':'data:image/jpeg;base64,'+base64.b64encode(encoded.tobytes()).decode()})])])
    request = urllib.request.Request(server_url, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    raw = payload['choices'][0]['message']['content']
    result = json.loads(raw)
    features = set(result.get('features',[]))
    valid_id = result.get('target_id') == target_id
    strong = result.get('confidence',0) >= .85
    result['passed'] = bool(valid_id and strong and result.get('person')=='yes'
        and 'torso' in features and bool(features & {'head','arm','leg'}))
    result['rejected'] = bool(valid_id and strong and result.get('person')=='no' and not features)
    return result, raw, payload.get('usage',{})

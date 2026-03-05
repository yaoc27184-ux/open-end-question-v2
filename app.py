#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开放题分析工具 - 通义千问AI增强版（Railway部署）
"""

from flask import Flask, request, send_file, jsonify, Response
import pandas as pd
import numpy as np
import re
import os
import json
import uuid
from datetime import datetime
import tempfile
import requests

app = Flask(__name__)
app.secret_key = 'open-question-analyzer-2024'

UPLOAD_FOLDER = tempfile.mkdtemp()
RESULTS_FOLDER = tempfile.mkdtemp()

# 通义千问 API 配置
DASHSCOPE_API_KEY = os.environ.get('DASHSCOPE_API_KEY', 'sk-ab126d5fcd534c6a9b63d9fedddbd342')

TRACK_MAP = {
    'Q16': '流畅性', 'Q17': '稳定性', 'Q18': '互联互通',
    'Q19': 'AI', 'Q20': '安全隐私', 'Q21': '美观性', 'Q22': '易用性'
}

SYSTEM_PROMPT = """你是一位资深的手机用户体验分析专家。请分析用户反馈，返回JSON格式结果。

规则：
1. 实质性判断：空白、纯符号为INVALID，有实际内容为VALID
2. 场景：从用户描述中提取使用场景，无法判断填N/A
3. APP/功能：提取涉及的APP或功能名称，无法判断填N/A
4. 问题类型：根据赛道选择对应的问题类型
5. 问题简述：一句话总结问题

问题类型参考：
- 流畅性：操作响应慢、掉帧/帧率不稳定、卡死/闪退、内容加载慢
- 稳定性：APP闪退/无响应、系统死机/重启、功耗/热异常、功能异常/bug
- 互联互通：连接慢/失败、找不到设备、步骤繁琐
- AI：AI结果质量差、AI意图理解不准、AI功能失败
- 安全隐私：隐私暴露风险、权限问题
- 美观性：视觉风格不统一、布局不协调
- 易用性：交互反直觉、入口难找"""

SCENE_KEYWORDS = {
    '打游戏': ['游戏', '玩', '对战', '团战', '王者', '和平精英', '原神'],
    '看视频': ['视频', '抖音', '刷', '哔哩哔哩', 'b站'],
    '日常使用': ['日常', '平时', '一般'],
    '启动/退出': ['启动', '打开', '退出', '进入'],
    '使用AI问答': ['ai', '小爱', '问答'],
    '连接(BT/WiFi)': ['连接', '蓝牙', 'wifi', '互联', '互传'],
}

APP_KEYWORDS = {
    '抖音': ['抖音'], '微信': ['微信'], '京东': ['京东'],
    '游戏': ['游戏', '玩游戏', '打游戏'], '小爱同学': ['小爱'],
    '小米互传/互联': ['互传', '互联'], '侧边栏': ['侧边栏'],
}

def call_qwen_api(prompt):
    """调用通义千问API"""
    if not DASHSCOPE_API_KEY:
        return None
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "qwen-turbo",
        "input": {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        },
        "parameters": {"result_format": "message"}
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        result = response.json()
        if 'output' in result and 'choices' in result['output']:
            return result['output']['choices'][0]['message']['content']
        print(f"API返回: {result}")
        return None
    except Exception as e:
        print(f"API调用失败: {e}")
        return None

def ai_analyze_batch(data_list, track):
    """AI批量分析"""
    if not data_list:
        return []
    prompt = f"""请分析以下「{track}」赛道的用户反馈，返回JSON数组：
{json.dumps(data_list, ensure_ascii=False)}
返回格式（只返回JSON数组）：
[{{"id": 原始ID, "scene": "场景", "app": "APP/功能", "problem_type": "问题类型", "summary": "问题简述"}}]"""
    result = call_qwen_api(prompt)
    if result:
        try:
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
    return []

def check_validity(text):
    if pd.isna(text) or text == '':
        return 'INVALID'
    text = str(text).strip()
    if text == '' or re.match(r'^[，。、；：！？\s\.,;:!?\-_]+$', text):
        return 'INVALID'
    return 'VALID'

def match_scene(text):
    text = str(text).lower()
    for scene, keywords in SCENE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return scene
    return 'N/A'

def match_app(text):
    text = str(text)
    for app, keywords in APP_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return app
    return 'N/A'

def match_problem_type(text, track):
    text = str(text).lower()
    if track == '流畅性':
        if '掉帧' in text or '帧' in text: return '掉帧/帧率不稳定'
        if '卡顿' in text or '卡' in text: return '操作响应慢'
        if '闪退' in text: return '卡死/闪退'
        return '操作响应慢'
    elif track == '稳定性':
        if '死机' in text or '重启' in text: return '系统死机/重启/开关机异常'
        if '闪退' in text: return 'APP闪退/无响应'
        if '发热' in text: return '功耗/热异常'
        return '功能异常/失效/bug'
    elif track == '互联互通':
        if '慢' in text: return '连接慢/失败'
        if '找不到' in text: return '找不到设备'
        return '连接慢/失败'
    elif track == 'AI':
        if '笨' in text or '不准' in text: return 'AI意图理解不准'
        return 'AI结果质量差'
    elif track == '安全隐私':
        return '保护方式不合理'
    elif track == '美观性':
        if '还行' in text: return 'N/A'
        return '视觉风格不统一'
    elif track == '易用性':
        if '还行' in text: return 'N/A'
        return '结果不可靠/不符合预期'
    return 'N/A'


def fallback_analysis(df_valid):
    """关键词匹配分析"""
    rows = []
    for _, row in df_valid.iterrows():
        text = str(row['用户原话'])
        track = row['赛道']
        parts = re.split(r'[；;。]', text)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 1]
        if len(parts) <= 1:
            parts = [text]
        for part in parts:
            rows.append({
                'ID': row['ID'], '题号': row['题号'], '赛道': track,
                '用户原话': part, '场景': match_scene(part), 'APP/功能': match_app(part),
                '问题类型': match_problem_type(part, track),
                '问题简述': part[:20] + '...' if len(part) > 20 else part
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['ID', '题号', '赛道', '用户原话', '场景', 'APP/功能', '问题类型', '问题简述'])

def run_analysis(filepath, result_dir):
    """执行分析"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
    
    questions = ['Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22']
    
    # 阶段零
    rows = []
    for _, row in df.iterrows():
        for q in questions:
            content = row[q] if q in row and pd.notna(row[q]) else ''
            rows.append({'ID': int(row['ID']), '题号': q, '用户原话': content})
    df_stage0 = pd.DataFrame(rows)
    df_stage0.to_excel(os.path.join(result_dir, '阶段零_原始数据重排.xlsx'), index=False)
    
    # 第一阶段
    rows = []
    for _, row in df_stage0.iterrows():
        text = row['用户原话']
        track = TRACK_MAP[row['题号']]
        validity = check_validity(text)
        rows.append({
            'ID': row['ID'], '题号': row['题号'], '赛道': track,
            '用户原话': text if pd.notna(text) else '',
            '实质性判断': validity, '赛道判断': 'INVALID' if validity == 'INVALID' else 'VALID'
        })
    df_stage1 = pd.DataFrame(rows)
    df_stage1.to_excel(os.path.join(result_dir, '第一阶段_数据清洗与赛道映射.xlsx'), index=False)
    
    # 第二阶段 - AI分析
    df_valid = df_stage1[(df_stage1['实质性判断'] == 'VALID') & (df_stage1['赛道判断'] == 'VALID')].copy()
    
    use_ai = bool(DASHSCOPE_API_KEY) and len(df_valid) > 0
    if use_ai:
        all_results = []
        for track in ['流畅性', '稳定性', '互联互通', 'AI', '安全隐私', '美观性', '易用性']:
            track_data = df_valid[df_valid['赛道'] == track]
            if len(track_data) == 0:
                continue
            data_list = [{'id': int(row['ID']), 'text': str(row['用户原话'])} for _, row in track_data.iterrows()]
            for i in range(0, len(data_list), 10):
                batch = data_list[i:i+10]
                results = ai_analyze_batch(batch, track)
                for r in results:
                    r['赛道'] = track
                    r['题号'] = [q for q, t in TRACK_MAP.items() if t == track][0]
                all_results.extend(results)
        
        if all_results:
            df_stage2 = pd.DataFrame(all_results)
            df_stage2 = df_stage2.rename(columns={'id': 'ID', 'scene': '场景', 'app': 'APP/功能', 'problem_type': '问题类型', 'summary': '问题简述'})
            id_text_map = dict(zip(df_valid['ID'].astype(str) + df_valid['赛道'], df_valid['用户原话']))
            df_stage2['用户原话'] = df_stage2.apply(lambda x: id_text_map.get(str(x['ID']) + x['赛道'], ''), axis=1)
        else:
            df_stage2 = fallback_analysis(df_valid)
    else:
        df_stage2 = fallback_analysis(df_valid)
    
    df_stage2.to_excel(os.path.join(result_dir, '第二阶段_结构化打标.xlsx'), index=False)
    
    # 第三阶段
    df_valid2 = df_stage2[df_stage2['问题类型'] != 'N/A'].copy() if '问题类型' in df_stage2.columns else df_stage2
    tracks = ['流畅性', '稳定性', '互联互通', 'AI', '安全隐私', '美观性', '易用性']
    track_questions = {'流畅性': 'Q16', '稳定性': 'Q17', '互联互通': 'Q18', 'AI': 'Q19', '安全隐私': 'Q20', '美观性': 'Q21', '易用性': 'Q22'}
    all_stats = []
    for track in tracks:
        track_data = df_valid2[df_valid2['赛道'] == track] if len(df_valid2) > 0 else pd.DataFrame()
        q_num = track_questions[track]
        for col, stat_type in [('场景', '场景统计'), ('APP/功能', 'APP/功能统计'), ('问题类型', '问题类型统计')]:
            if len(track_data) > 0 and col in track_data.columns:
                stats = track_data.groupby(col).size().reset_index(name='数量')
            else:
                stats = pd.DataFrame(columns=[col, '数量'])
            stats['赛道'] = f"{q_num} {track}"
            stats['统计类型'] = stat_type
            stats = stats.rename(columns={col: '维度'})
            if len(stats) == 0:
                stats = pd.DataFrame([{'赛道': f"{q_num} {track}", '统计类型': stat_type, '维度': 'N/A', '数量': 0}])
            all_stats.append(stats)
    df_stage3 = pd.concat(all_stats, ignore_index=True)[['赛道', '统计类型', '维度', '数量']]
    df_stage3.to_excel(os.path.join(result_dir, '第三阶段_量化统计.xlsx'), index=False)
    
    return generate_report_data(df_stage2, df_stage3, len(df), len(df_valid), len(df_stage2))

def generate_report_data(df_stage2, df_stage3, total, valid, tagged):
    df_valid = df_stage2[df_stage2['问题类型'] != 'N/A'].copy() if '问题类型' in df_stage2.columns and len(df_stage2) > 0 else pd.DataFrame()
    tracks = ['流畅性', '稳定性', '互联互通', 'AI', '安全隐私', '美观性', '易用性']
    track_questions = {'流畅性': 'Q16', '稳定性': 'Q17', '互联互通': 'Q18', 'AI': 'Q19', '安全隐私': 'Q20', '美观性': 'Q21', '易用性': 'Q22'}
    report = []
    for track in tracks:
        q_num = track_questions[track]
        track_data = df_valid[df_valid['赛道'] == track] if len(df_valid) > 0 else pd.DataFrame()
        track_stats = df_stage3[df_stage3['赛道'] == f"{q_num} {track}"] if len(df_stage3) > 0 else pd.DataFrame()
        scene_stats = track_stats[track_stats['统计类型'] == '场景统计'].sort_values('数量', ascending=False).to_dict('records') if len(track_stats) > 0 else []
        app_stats = track_stats[track_stats['统计类型'] == 'APP/功能统计'].sort_values('数量', ascending=False).to_dict('records') if len(track_stats) > 0 else []
        problem_stats = track_stats[track_stats['统计类型'] == '问题类型统计'].sort_values('数量', ascending=False).to_dict('records') if len(track_stats) > 0 else []
        topics = []
        if len(track_data) > 0:
            for problem_type, group in track_data.groupby('问题类型'):
                if len(topics) >= 5: break
                quotes = group['用户原话'].head(5).tolist()
                topics.append({'name': problem_type, 'quotes': quotes})
        top_problem = problem_stats[0]['维度'] if problem_stats and problem_stats[0]['数量'] > 0 else 'N/A'
        total_p = sum(p['数量'] for p in problem_stats)
        ratio = (problem_stats[0]['数量'] / total_p * 100) if total_p > 0 and problem_stats else 0
        report.append({
            'track': track, 'q_num': q_num, 'total': len(track_data),
            'scene_stats': scene_stats, 'app_stats': app_stats, 'problem_stats': problem_stats,
            'topics': topics, 'top_problem': top_problem, 'top_ratio': round(ratio, 1),
            'conclusion': f"{track}赛道问题主要表现为「{top_problem}」，占比{ratio:.1f}%，建议针对性优化。" if top_problem != 'N/A' else f"{track}赛道有效反馈较少。"
        })
    return {'total_records': total, 'valid_records': valid, 'tagged_records': tagged, 'report': report}


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>开放题分析工具 - AI版</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Segoe UI", sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: #fff; padding: 40px 0; }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 1.1em; }
        .ai-badge { display: inline-block; background: #28a745; color: #fff; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin-left: 10px; }
        .upload-card { background: #fff; border-radius: 16px; padding: 40px; box-shadow: 0 20px 60px rgba(0,0,0,0.3); margin-bottom: 30px; }
        .upload-zone { border: 3px dashed #667eea; border-radius: 12px; padding: 60px; text-align: center; cursor: pointer; transition: all 0.3s; background: #f8f9ff; }
        .upload-zone:hover { border-color: #764ba2; background: #f0f2ff; }
        .upload-zone.dragover { border-color: #28a745; background: #e8f5e9; }
        .upload-icon { font-size: 4em; margin-bottom: 20px; }
        .upload-text { color: #666; font-size: 1.2em; }
        .upload-hint { color: #999; margin-top: 10px; }
        #fileInput { display: none; }
        .btn { background: linear-gradient(90deg, #667eea, #764ba2); color: #fff; border: none; padding: 15px 40px; border-radius: 8px; font-size: 1.1em; cursor: pointer; margin-top: 20px; transition: transform 0.2s; }
        .btn:hover { transform: scale(1.05); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .progress { display: none; margin-top: 30px; }
        .progress-bar { height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); width: 0%; transition: width 0.3s; }
        .progress-text { text-align: center; margin-top: 10px; color: #666; }
        .download-btn { display: block; padding: 15px; background: #f8f9fa; border-radius: 8px; text-decoration: none; color: #333; text-align: center; transition: all 0.2s; margin-bottom: 10px; }
        .download-btn:hover { background: #667eea; color: #fff; }
        .stats-summary { display: flex; justify-content: space-around; margin: 30px 0; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; }
        .stat-item { text-align: center; color: #fff; }
        .stat-num { display: block; font-size: 2.5em; font-weight: bold; }
        .stat-label { opacity: 0.9; }
        .track-section { margin-top: 30px; border-top: 2px solid #eee; padding-top: 20px; }
        .track-section h3 { background: linear-gradient(90deg, #667eea, #764ba2); color: #fff; padding: 12px 20px; border-radius: 8px; margin-bottom: 20px; }
        .track-count { font-weight: normal; opacity: 0.8; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .stats-block { background: #f8f9fa; padding: 15px; border-radius: 8px; }
        .stats-block h4 { margin-bottom: 10px; color: #333; }
        .stats-block table { width: 100%; font-size: 0.9em; }
        .stats-block th { background: #343a40; color: #fff; padding: 8px; text-align: left; }
        .stats-block td { padding: 6px 8px; border-bottom: 1px solid #ddd; }
        .topics { margin-top: 20px; }
        .topics h4 { margin-bottom: 15px; }
        .topic { background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
        .topic-title { font-weight: bold; color: #667eea; margin-bottom: 10px; }
        .topic ul { padding-left: 20px; }
        .topic li { margin: 5px 0; color: #555; }
        .conclusion { background: linear-gradient(135deg, #e8f4f8 0%, #d4edda 100%); padding: 15px 20px; border-radius: 8px; margin-top: 20px; border-left: 4px solid #28a745; font-weight: 500; }
        @media (max-width: 768px) { .stats-grid { grid-template-columns: 1fr; } .header h1 { font-size: 1.8em; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header"><h1>📊 开放题分析工具 <span class="ai-badge">🤖 通义千问AI</span></h1><p>上传 NPS/VOC 数据，AI 自动生成深度分析报告</p></div>
        <div class="upload-card">
            <div class="upload-zone" id="uploadZone"><div class="upload-icon">📁</div><div class="upload-text">点击或拖拽上传 Excel 文件</div><div class="upload-hint">支持 .xlsx, .xls, .csv 格式</div></div>
            <input type="file" id="fileInput" accept=".xlsx,.xls,.csv">
            <div style="text-align: center;"><button class="btn" id="analyzeBtn" disabled>🚀 开始AI分析</button></div>
            <div class="progress" id="progress"><div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div><div class="progress-text" id="progressText">AI正在分析中...</div></div>
        </div>
        <div id="results"></div>
    </div>
<script>
const uploadZone = document.getElementById('uploadZone'), fileInput = document.getElementById('fileInput'), analyzeBtn = document.getElementById('analyzeBtn'), progress = document.getElementById('progress'), progressFill = document.getElementById('progressFill'), progressText = document.getElementById('progressText'), results = document.getElementById('results');
let selectedFile = null;
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', (e) => { e.preventDefault(); uploadZone.classList.remove('dragover'); if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });
function handleFile(file) { selectedFile = file; uploadZone.innerHTML = '<div class="upload-icon">✅</div><div class="upload-text">'+file.name+'</div><div class="upload-hint">点击重新选择</div>'; analyzeBtn.disabled = false; }
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    analyzeBtn.disabled = true; progress.style.display = 'block'; results.innerHTML = '';
    let prog = 0; const interval = setInterval(() => { prog = Math.min(prog + Math.random() * 8, 90); progressFill.style.width = prog + '%'; progressText.textContent = 'AI正在分析中... ' + Math.round(prog) + '%'; }, 1000);
    const formData = new FormData(); formData.append('file', selectedFile);
    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const data = await response.json();
        clearInterval(interval); progressFill.style.width = '100%'; progressText.textContent = '✅ 分析完成！';
        if (data.error) { results.innerHTML = '<div class="upload-card" style="color:red;">❌ 错误: '+data.error+'</div>'; }
        else { renderResults(data); }
    } catch (err) { clearInterval(interval); results.innerHTML = '<div class="upload-card" style="color:red;">❌ 请求失败: '+err.message+'</div>'; }
    analyzeBtn.disabled = false;
});
function renderResults(data) {
    let html = '<div class="upload-card"><h2 style="margin-bottom:20px;">📥 下载分析结果</h2><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;"><a href="/download/'+data.task_id+'/阶段零_原始数据重排.xlsx" class="download-btn">📊 阶段零_原始数据重排</a><a href="/download/'+data.task_id+'/第一阶段_数据清洗与赛道映射.xlsx" class="download-btn">🧹 第一阶段_数据清洗</a><a href="/download/'+data.task_id+'/第二阶段_结构化打标.xlsx" class="download-btn">🏷️ 第二阶段_结构化打标</a><a href="/download/'+data.task_id+'/第三阶段_量化统计.xlsx" class="download-btn">📈 第三阶段_量化统计</a></div></div><div class="upload-card"><h2 style="margin-bottom:20px;">📝 深度分析报告</h2><div class="stats-summary"><div class="stat-item"><span class="stat-num">'+data.total_records+'</span><span class="stat-label">原始数据</span></div><div class="stat-item"><span class="stat-num">'+data.valid_records+'</span><span class="stat-label">有效数据</span></div><div class="stat-item"><span class="stat-num">'+data.tagged_records+'</span><span class="stat-label">打标数据</span></div></div>';
    data.report.forEach(track => {
        html += '<div class="track-section"><h3>'+track.q_num+' '+track.track+' <span class="track-count">('+track.total+'条)</span></h3><div class="track-content"><div class="stats-grid"><div class="stats-block"><h4>场景分布</h4><table><tr><th>场景</th><th>数量</th></tr>'+track.scene_stats.map(s => '<tr><td>'+s.维度+'</td><td>'+s.数量+'</td></tr>').join('')+'</table></div><div class="stats-block"><h4>APP/功能分布</h4><table><tr><th>APP/功能</th><th>数量</th></tr>'+track.app_stats.map(s => '<tr><td>'+s.维度+'</td><td>'+s.数量+'</td></tr>').join('')+'</table></div><div class="stats-block"><h4>问题类型分布</h4><table><tr><th>问题类型</th><th>数量</th></tr>'+track.problem_stats.map(s => '<tr><td>'+s.维度+'</td><td>'+s.数量+'</td></tr>').join('')+'</table></div></div><div class="topics"><h4>话题归纳</h4>'+track.topics.map((t, i) => '<div class="topic"><div class="topic-title">话题'+(i+1)+'：'+t.name+'</div><ul>'+t.quotes.map(q => '<li>「'+q+'」</li>').join('')+'</ul></div>').join('')+'</div><div class="conclusion">💡 '+track.conclusion+'</div></div></div>';
    });
    html += '</div>'; results.innerHTML = html;
}
</script>
</body></html>'''

@app.route('/')
def index():
    return Response(HTML_TEMPLATE, mimetype='text/html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    task_id = str(uuid.uuid4())
    result_dir = os.path.join(RESULTS_FOLDER, task_id)
    os.makedirs(result_dir, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, f"{task_id}_{file.filename}")
    file.save(filepath)
    try:
        result = run_analysis(filepath, result_dir)
        result['task_id'] = task_id
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<task_id>/<filename>')
def download(task_id, filename):
    filepath = os.path.join(RESULTS_FOLDER, task_id, filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': '文件不存在'}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)

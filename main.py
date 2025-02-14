import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np


# 初始化会话状态
def init_session_state():
    session_defaults = {
        'conversation': [],
        'user_profile': {
            'name': '',
            'age': 30,
            'gender': 'male',
            'medical_history': []
        },
        'diagnosis_data': {},
        'medications': [],
        'audio_buffer': []
    }

    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# 语音处理回调
def audio_callback(frame: av.AudioFrame) -> av.AudioFrame:
    audio_data = frame.to_ndarray()
    st.session_state.audio_buffer.append(audio_data)
    return frame


# 语音输入组件
def voice_input_component():
    with st.expander("语音输入", expanded=False):
        ctx = webrtc_streamer(
            key="voice",
            mode=av.AudioOnly,
            audio_frame_callback=audio_callback,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        if ctx.state.playing and st.button("结束录音"):
            audio_array = np.concatenate(st.session_state.audio_buffer)
            text = speech_to_text(audio_array)  # 需接入实际ASR服务
            if text:
                process_user_input(text)
            st.session_state.audio_buffer = []


# 可视化组件
def visualization_components():
    with st.expander("诊断报告", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            render_symptom_network()
        with col2:
            render_diagnosis_report()


# 症状关联图谱
def render_symptom_network():
    data = st.session_state.diagnosis_data.get('symptom_network', {})
    if data:
        df = pd.DataFrame({
            'source': [d['source'] for d in data['links']],
            'target': [d['target'] for d in data['links']],
            'value': [d['value'] for d in data['links']]
        })

        fig = px.scatter(df, x='source', y='target', size='value',
                         title="症状关联图谱",
                         labels={'source': '当前症状', 'target': '关联症状'},
                         height=300)
        st.plotly_chart(fig, use_container_width=True)


# 疾病概率分布
def render_diagnosis_report():
    data = st.session_state.diagnosis_data.get('probabilities', {})
    if data:
        df = pd.DataFrame({
            'disease': list(data.keys()),
            'probability': list(data.values())
        })
        fig = px.bar(df, x='disease', y='probability',
                     title="疾病概率分布",
                     color='probability',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)


# 用药提醒模块
def medication_reminder():
    with st.sidebar.expander("💊 用药提醒", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            new_med = st.text_input("药品名称", key="new_med")
        with col2:
            med_time = st.time_input("提醒时间", key="med_time")

        if st.button("添加提醒"):
            st.session_state.medications.append({
                'name': new_med,
                'time': med_time.strftime("%H:%M"),
                'status': 'pending'
            })

        st.divider()
        for idx, med in enumerate(st.session_state.medications):
            cols = st.columns([1, 2, 1])
            cols[0].write(f"⏰ {med['time']}")
            cols[1].write(med['name'])
            if cols[2].button("✓", key=f"med_{idx}"):
                st.session_state.medications[idx]['status'] = 'done'

            if med['status'] == 'done':
                st.markdown("---")


# 用户档案管理
def user_profile_section():
    with st.sidebar:
        st.subheader("📁 用户档案")
        name = st.text_input("姓名", value=st.session_state.user_profile['name'])
        age = st.number_input("年龄", min_value=1, max_value=100,
                              value=st.session_state.user_profile['age'])
        gender = st.selectbox("性别", ['男', '女'],
                              index=0 if st.session_state.user_profile['gender'] == 'male' else 1)

        if st.button("更新档案"):
            st.session_state.user_profile.update({
                'name': name,
                'age': age,
                'gender': gender
            })
            st.success("档案已更新")


# 对话界面
def chat_interface():
    st.header("🏥 智能医疗问诊系统")

    # 主界面布局
    col1, col2 = st.columns([3, 2])

    with col1:
        # 对话历史
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.conversation:
                bubble_style = """
                    padding: 10px;
                    border-radius: 15px;
                    margin: 5px 0;
                    max-width: 80%;
                """
                if msg['type'] == 'user':
                    st.markdown(f"""
                        <div style="{bubble_style} 
                            background: #e3f2fd; 
                            margin-left: auto;
                            text-align: right;">
                            {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="{bubble_style} 
                            background: #f5f5f5;
                            margin-right: auto;">
                            {msg['content']}
                        </div>
                    """, unsafe_allow_html=True)

        # 输入区域
        input_col, voice_col, btn_col = st.columns([3, 1, 1])
        with input_col:
            user_input = st.text_input("请输入症状描述", key="user_input",
                                       placeholder="例如：头痛三天，伴有发热",
                                       label_visibility="collapsed")
        with voice_col:
            voice_input_component()
        with btn_col:
            if st.button("发送", use_container_width=True, type="primary"):
                process_user_input(user_input)
                st.session_state.user_input = ""  # 清空输入框

    with col2:
        # 预警系统
        if st.session_state.get('warning'):
            st.error(f"⚠️ 高危症状预警：{st.session_state.warning}")

        # 可视化组件
        visualization_components()


# 处理用户输入
def process_user_input(text):
    if text.strip():
        # 记录用户输入
        st.session_state.conversation.append({
            'type': 'user',
            'content': text,
            'time': datetime.now().strftime("%H:%M:%S")
        })

        # 获取AI响应（模拟）
        response = get_ai_response(text)

        # 记录系统响应
        st.session_state.conversation.append({
            'type': 'system',
            'content': response['answer'],
            'time': datetime.now().strftime("%H:%M:%S")
        })

        # 更新诊断数据
        st.session_state.diagnosis_data = response.get('data', {})
        st.session_state.warning = response.get('warning')


# 模拟AI服务（需替换为实际API调用）
def get_ai_response(text):
    # 此处应调用FastAPI后端
    return {
        "answer": f"收到症状：{text}。请补充说明：疼痛程度如何？",
        "data": {
            "symptom_network": {
                "nodes": [
                    {"id": "头痛", "group": 1},
                    {"id": "发热", "group": 2}
                ],
                "links": [
                    {"source": "头痛", "target": "发热", "value": 0.8}
                ]
            },
            "probabilities": {
                "感冒": 0.65,
                "流感": 0.25,
                "偏头痛": 0.1
            }
        },
        "warning": "持续高热需立即就医" if "发热" in text else None
    }


# 主函数
def main():
    st.set_page_config(
        page_title="智能医疗问诊系统",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    user_profile_section()
    medication_reminder()
    chat_interface()


if __name__ == "__main__":
    main()
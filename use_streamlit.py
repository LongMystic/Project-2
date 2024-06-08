# import lib
import os
import time
import re
import openai
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings.openai import OpenAIEmbeddings
from openai import AsyncOpenAI
import asyncio
from streamlit_star_rating import st_star_rating
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = 'llm-chatbot-gpt'
embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pVS = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding
)

primer2 = f"""Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn hãy tóm tắt tình huống
và đưa ra các câu hỏi ngắn gọn gồm từ khoá liên quan tới pháp luật được suy luận từ phần thông tin có trong tình huống. Các câu trả lời của bạn
đều là tiếng việt.

Sau đây là một số ví dụ và phần tóm tắt:
1. Tình huống: Biết mình đủ tuổi kết hôn và đáp ứng các điều kiện kết hôn, Anh S và chị Y dự định đi đăng ký kết hôn trước khi tổ chức lễ cưới 02 tháng. Chị Y và anh S có hộ khẩu thường trú ở hai tỉnh khác nhau, anh chị muốn biết việc đăng ký kết hôn thực hiện tại cơ quan nào và cần thực hiện thủ tục gì?
-> Tóm tắt: Thủ tục đăng ký kết hôn là gì, hộ khẩu thường trú trong thủ tục kết hôn

2. Ông bà B có con trai đã 25 tuổi, bị bệnh đao bẩm sinh. Vì muốn lấy vợ cho con trai, bà B đã tìm cách vu cáo cho chị Y – người giúp việc lấy trộm số tiền 1.000.000 đồng. Bà B  đe dọa nếu chị Y không muốn bị báo công an, không muốn bị đi tù thì phải lấy con trai bà, vừa được làm chủ nhà, không phải làm người giúp việc lại có cuộc sống sung túc. Vì nhận thức hạn chế, trình độ văn hóa thấp nên chị Y đã đồng ý lấy con trai bà B. Hôn lễ chỉ tổ chức giữa hai gia đình mà không làm thủ tục đăng ký kết hôn tại phường. Việc làm của bà B có vi phạm pháp luật không? Nếu có thì bị xử phạt như thế nào?
-> Tóm tắt: cưỡng ép kết hôn có bị xử phạt không, cưỡng ép kết hôn bị xử phạt như thế nào 

3. Tôi đã kết hôn được 6 tháng, nhưng chưa chuyển hộ khẩu về nhà chồng (ở xã X, huyện B, tỉnh A), hộ khẩu của tôi vẫn đang ở nhà bố mẹ đẻ (xã Y, huyện C, tỉnh D). Nay tôi có nguyện vọng chuyển hộ khẩu về nhà chồng thì có được không và thủ tục thực hiện như thế nào? Ai có thẩm quyền giải quyết?
-> tóm tắt: có được chuyển hộ khẩu về nhà chồng không, Thủ tục chuyển hộ khẩu, Ai giải quyết thủ tục chuyển hộ khẩu
"""

primer1 = f"""Bây giờ bạn hãy đóng vai trò là một luật sư xuất sắc về luật hôn nhân và gia đình ở Việt Nam.
Tôi sẽ hỏi bạn các câu hỏi về tình huống thực tế liên quan tới luật hôn nhân và gia đình. Bạn sẽ trả lời câu hỏi
dựa trên thông tin tôi cung cấp và thông tin có trong câu hỏi. Nếu thông tin tôi cung cấp không đủ để trả lời
hãy nói rằng 'Tôi không biết'. Các câu trả lời của bạn đều là tiếng việt. 
Lưu ý: Nêu rõ điều luật số mấy để trả lời tình huống.
"""

def check_string(s):
    pattern = r"Nếu bạn cần (.*) thông tin"
    match = re.search(pattern, s)
    if match:
        return True
    else:
        return False

def remove_string(s):
    pattern = r"Nếu bạn cần .*"
    return re.sub(pattern, "", s)
async def page_1():
    st.title("🧑‍💻💬 A RAG chatbot for family and marriage legal questions")
    """
    Đây là chatbot giúp người dân tìm hiểu luật hôn nhân và gia đình. Bạn hãy hỏi những câu hỏi có liên quan tới luật này nhé.
    """
    for conversation in st.session_state.chat_history1:
        st.chat_message("user").write(conversation['question'])
        st.chat_message("assisstant").write(conversation['answer'])
        st.write("Nếu cần tư vấn rõ hơn, bạn có thể liên hệ luật sư qua trang web [Luật minh khuê](https://luatminhkhue.vn/)")
        st.write(f"You rated this answer {conversation['star']} :star:")
    if "messages1" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": primer1}
        ]

    if "stars1" not in st.session_state:
        st.session_state.stars1 = ""

    if st.session_state.stars1:
        st.session_state.chat_history1.append(
            {'question': st.session_state['question1'],
             'answer': st.session_state['msg1'],
             'star': st.session_state['stars1']
             }
        )
        st.chat_message("user").write(st.session_state['question1'])
        st.chat_message("assistant").write(st.session_state['msg1'])
        st.write("Nếu cần tư vấn rõ hơn, bạn có thể liên hệ luật sư qua trang web [Luật minh khuê](https://luatminhkhue.vn/)")
        st.write(f"You rated this answer {st.session_state['stars1']} :star:")
        del st.session_state.prompt1

    if prompt1 := st.chat_input():
        st.session_state.prompt1 = prompt1
        st.chat_message("user").write(prompt1)
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": primer1},
                {"role": "user", "content": prompt1}
            ]
        )
        msg = response.choices[0].message.content
        st.chat_message("assistant").write(msg)
        st.session_state.question1 = prompt1
        st.session_state.msg1 = msg
        st.write("Nếu cần tư vấn rõ hơn, bạn có thể liên hệ luật sư qua trang web [Luật minh khuê](https://luatminhkhue.vn/)")

    if "prompt1" in st.session_state:
        stars1 = st_star_rating("Please rate you experience", maxValue=5, defaultValue=3, key="stars1")

async def page_2():
    st.title("🧑‍💻💬 A RAG chatbot for family and marriage legal questions")
    """
    Đây là chatbot giúp người dân tìm hiểu luật hôn nhân và gia đình. Bạn hãy hỏi những câu hỏi có liên quan tới luật này nhé.
    """
    for conversation in st.session_state.chat_history2:
        st.chat_message("user").write(conversation['question'])
        st.chat_message("assisstant").write(conversation['answer'])
        st.write("Nếu cần tư vấn rõ hơn, bạn có thể liên hệ luật sư qua trang web [Luật minh khuê](https://luatminhkhue.vn/)")
        st.write(f"You rated this answer {conversation['star']}")
    if "stars2" not in st.session_state:
        st.session_state.stars2 = ""

    if st.session_state.stars2:
        st.session_state.chat_history2.append(
            {'question': st.session_state['question'],
             'answer': st.session_state['msg'],
             'star': st.session_state['stars2']
             }
        )
        st.chat_message("user").write(st.session_state['question'])
        st.chat_message("assistant").write(st.session_state['msg'])
        st.write("Nếu cần tư vấn rõ hơn, bạn có thể liên hệ luật sư qua trang web [Luật minh khuê](https://luatminhkhue.vn/)")
        st.write(f"You rated this answer {st.session_state['stars2']} :star:")
        del st.session_state.prompt
    if prompt := st.chat_input():
        st.session_state.prompt = prompt
        st.chat_message("user").write(prompt)
        context = ""
        if len(prompt) <= 150:
            relevant_docs = pVS.similarity_search(query=prompt, k=1)
            context += relevant_docs[0].page_content
        else:
            chats = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": primer2},
                    {"role": "user", "content": prompt}
                ]
            )

            new_query = chats.choices[0].message.content
            context = ""

            texts = new_query.split(',')

            for text in texts:
                print(text)
                relevant_docs = pVS.similarity_search(query=text, k=1)

                context += relevant_docs[0].page_content

            relevant_docs = pVS.similarity_search(query=prompt, k=1)
            context += relevant_docs[0].page_content
        context += "\n" + prompt

        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": primer1},
                {"role": "user", "content": context}
            ]
        )
        msg = response.choices[0].message.content
        if check_string(msg):
            msg = remove_string(msg)
        st.chat_message("assistant").write(msg)
        st.write("Nếu cần tư vấn rõ hơn, bạn có thể liên hệ luật sư qua trang web [Luật minh khuê](https://luatminhkhue.vn/)")
        st.session_state.question = prompt
        st.session_state.msg = msg

    if "prompt" in st.session_state:
        stars2 = st_star_rating("Please rate you experience", maxValue=5, defaultValue=3, key="stars2")

PAGES = {
    "GPT-3.5-Turbo": page_1,
    "GPT-3.5-Turbo + RAG": page_2
}

def main():
    if "chat_history1" not in st.session_state:
        st.session_state.chat_history1 = []
    if "chat_history2" not in st.session_state:
        st.session_state.chat_history2 = []
    st.set_page_config(page_title="RAG Chatbot")
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select Chatbot", list(PAGES.keys()))
    # Call the page function
    asyncio.run(PAGES[choice]())

if __name__ == "__main__":
    main()






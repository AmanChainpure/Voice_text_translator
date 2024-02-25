import streamlit as st
import os
import pandas as pd
import PIL.Image
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.palm import PaLM
from llama_index import ServiceContext, load_index_from_storage
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM, OpenAI
from llama_index import set_global_service_context
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from llama_index.query_engine.pandas import PandasInstructionParser
from llama_index.prompts import PromptTemplate
from llama_index.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)

def upload_file():
    uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
    if uploaded_file is not None:
        st.success("File successfully uploaded")
        return uploaded_file.getvalue()  # Return file content as bytes
    return None

def ask_question(query_engine, response):
    question = st.text_input("Ask your question")
    if st.button("Submit"):
        if question:
            response_obj = query_engine.query(question)
            response_text = response_obj.response
            st.write("Response:", response_text)
        else:
            st.warning("Please enter a question")


def main():
    st.sidebar.title("Tool Box")
    tool_selection = st.sidebar.radio("Select Tool", ["Unstructured", "Image", "Structured"])
    if tool_selection == "Unstructured":
        st.sidebar.title("Unstructured Data QA")
        st.sidebar.write("Upload your PDF file and ask your questions.")

        os.environ['GOOGLE_API_KEY'] = "GOOGLE_API_KEY"
        os.environ['GRADIENT_ACCESS_TOKEN'] = 'GRADIENT_ACCESS_TOKEN'
        os.environ['GRADIENT_WORKSPACE_ID'] = 'GRADIENT_WORKSPACE_ID'

        llm = PaLM()

        embed_model = GradientEmbedding(
            gradient_access_token=os.environ["GRADIENT_ACCESS_TOKEN"],
            gradient_workspace_id=os.environ["GRADIENT_WORKSPACE_ID"],
            gradient_model_slug="bge-large",
        )

        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed_model,
            chunk_size=256,
        )

        set_global_service_context(service_context)

        index = VectorStoreIndex.from_documents(
            SimpleDirectoryReader("data").load_data(),
            service_context=service_context
        )

        query_engine = index.as_query_engine()


        uploaded_file = upload_file()
        if uploaded_file:
            ask_question(query_engine, uploaded_file)

    elif tool_selection == "Image":
        st.title("Image based QA")

        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            st.success("Image uploaded successfully!")
            image = PIL.Image.open(uploaded_file)
            # Display uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            question = st.text_input("Ask a question about the image", "What's in this image?")

            if st.button("Generate Response"):
                os.environ['GOOGLE_API_KEY'] = "GOOGLE_API_KEY"

                llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")

                hmessage1 = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": question,
                        },
                        {"type": "image_url", "image_url": image},
                    ]
                )
                message1 = llm.invoke([hmessage1])

                st.write("Response:", message1.content)

    elif tool_selection == "Structured":
        st.sidebar.title("Structured Data QA")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            st.sidebar.success("File successfully uploaded")
            df = pd.read_csv(uploaded_file)
            instruction_str = (
                "1. Convert the query to executable Python code using Pandas.\n"
                "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
                "3. The code should represent a solution to the query.\n"
                "4. PRINT ONLY THE EXPRESSION.\n"
                "5. Do not quote the expression.\n"
            )
            pandas_prompt_str = (
                "You are working with a pandas dataframe in Python.\n"
                "The name of the dataframe is `df`.\n"
                "This is the result of `print(df.head())`:\n"
                "{df_str}\n\n"
                "Follow these instructions:\n"
                "{instruction_str}\n"
                "Query: {query_str}\n\n"
                "Expression:"
            )
            response_synthesis_prompt_str = (
                "Given an input question, synthesize a response from the query results.\n"
                "Query: {query_str}\n\n"
                "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
                "Pandas Output: {pandas_output}\n\n"
                "Response: "
            )
            pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
                instruction_str=instruction_str, df_str=df.head(5)
            )
            pandas_output_parser = PandasInstructionParser(df)
            response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
            llm = OpenAI(model="gpt-4")
            qp = QP(
                modules={
                    "input": InputComponent(),
                    "pandas_prompt": pandas_prompt,
                    "llm1": llm,
                    "pandas_output_parser": pandas_output_parser,
                    "response_synthesis_prompt": response_synthesis_prompt,
                    "llm2": llm,
                },
                verbose=True,
            )
            qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
            qp.add_links(
                [
                    Link("input", "response_synthesis_prompt", dest_key="query_str"),
                    Link(
                        "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
                    ),
                    Link(
                        "pandas_output_parser",
                        "response_synthesis_prompt",
                        dest_key="pandas_output",
                    ),
                ]
            )
            qp.add_link("response_synthesis_prompt", "llm2")
            user_query = st.text_input("Ask your question:")
            if st.button("Get Response"):
                response = qp.run(query_str=user_query)
                st.write("Response:", response.message.content)
if __name__ == "__main__":
    main()


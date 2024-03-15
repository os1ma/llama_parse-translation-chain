import logging
import os
import sys
from collections.abc import Generator

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_parse import LlamaParse  # type: ignore[import-untyped]

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIR = "./tmp"


def parse_pdf(pdf_file: str) -> str:
    """
    Ref: https://github.com/run-llama/llama_parse
    """
    parser = LlamaParse(
        result_type="markdown",  # "markdown" and "text" are available
        num_workers=4,  # if multiple files passed, split in `num_workers` API calls
        verbose=True,
        language="en",  # Optionaly you can define a language, default=en
    )

    logger.info("Parsing PDF: %s", pdf_file)
    documents = parser.load_data(pdf_file)
    logger.info("Parsed %d documents", len(documents))
    texts = [doc.text for doc in documents]
    return "\n".join(texts)


def split_markdown(text: str) -> list[str]:
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    text_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    split_docs = text_splitter.split_text(text)
    return [doc.page_content for doc in split_docs]


def translate_doc(few_shot: list[BaseMessage] | None, text: str) -> str:
    llm = ChatOpenAI(
        model=os.environ["OPENAI_API_MODEL"],
        temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Translate English to Japanese. Don't generate new content."),
            MessagesPlaceholder(variable_name="few_shot", optional=True),
            ("user", "{text}"),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"few_shot": few_shot, "text": text})


MAX_FEW_SHOT = 3


def translate(texts: list[str]) -> Generator[str, None, None]:
    few_shot: list[BaseMessage] = []

    for text in texts:
        prev_output = translate_doc(few_shot, text)

        few_shot.append(HumanMessage(content=text))
        few_shot.append(AIMessage(content=prev_output))

        if len(few_shot) > MAX_FEW_SHOT * 2:
            few_shot = few_shot[2:]

        yield prev_output


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    parsed_file = f"{PERSIST_DIR}/{pdf_file}.txt"
    translated_file = f"{PERSIST_DIR}/{pdf_file}.ja.txt"

    exist_parsed_file = os.path.exists(parsed_file)
    if exist_parsed_file:
        logger.info("Reading parsed file: %s", parsed_file)
        text = open(parsed_file).read()
    else:
        text = parse_pdf(pdf_file)
        with open(parsed_file, "w") as f:
            f.write(text)

    split_texts = split_markdown(text)

    with open(translated_file, "w") as f:
        for chunk in translate(split_texts):
            f.write(f"{chunk}\n")


main()

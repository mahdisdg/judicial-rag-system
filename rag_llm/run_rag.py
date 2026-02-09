from src.rag_pipeline import RAGPipeline

def main():
    rag = RAGPipeline()
    query = input("سوال حقوقی خود را وارد کنید:\n")
    result = rag.run(query)

    print("\nپاسخ:\n")
    print(result["answer"])

    print("\nاستنادها:")
    for c in result["citations"]:
        print("-", c)

if __name__ == "__main__":
    main()

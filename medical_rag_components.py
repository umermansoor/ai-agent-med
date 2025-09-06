
"""
Medical RAG - Individual Components
An Example showing retriever, grader, rewriter, and answer generator for medical Q&A.
"""



# Load shared configuration
import config

# Import components
from grader import grade_documents
from rewriter import rewrite_question
from generate_answer import generate_answer
from retriever import create_retriever

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools.retriever import create_retriever_tool

def test_medical_workflow():
    """Test the complete medical RAG workflow."""
    retriever = create_retriever("drapoel")
    
    # Test question
    original_question = "what prescription medications is the patient taking?"
    
    print(f"\n� Original Question: '{original_question}'")
    
    # Step 1: Rewrite question for better retrieval
    rewriter_state = {"messages": [HumanMessage(content=original_question)]}
    rewritten_state = rewrite_question(rewriter_state)
    improved_question = rewritten_state["messages"][0]["content"]
    
    print(f"\n\n🔄 Improved Question: '{improved_question}'")
    
    # Step 2: Retrieve relevant documents
    docs = retriever.invoke(improved_question)
    print(f"\n\n📊 Retrieved {len(docs)} document chunks")
    
    # Combine all retrieved documents for context
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else "No documents found"
    
    print(f"\n\n📄 Combined Context: {context[:200]}...")
    
    # Step 3: Grade document relevance
    grader_state = {
        "messages": [
            HumanMessage(content=improved_question),
            AIMessage(content=context)
        ]
    }
    decision = grade_documents(grader_state)
    print(f"\n\n⚖️ Grader Decision: {decision}")
    
    # Step 4: Generate answer if documents are relevant
    if decision == "generate_answer":
        answer_state = generate_answer(grader_state)
        medical_answer = answer_state["messages"][0].content
        print(f"\n\n🏥 Medical Answer: {medical_answer}")
    else:
        print("\n\n📝 Documents not relevant - would rewrite question again")


def test_question_rewriting():
    """Test the question rewriting component."""
    
    sample_questions = [
        "does the patient have diabetes?",
        "list the patient's current medications",
        "the person reported feeling fatigued and weak. what could be the cause?",
        "what's the health status of the patient?",
    ]

    for question in sample_questions:
        state = {"messages": [HumanMessage(content=question)]}
        rewritten_state = rewrite_question(state)
        improved_question = rewritten_state["messages"][0]["content"]
        print(f"\n� Original Question: '{question}'")
        print(f"🔄 Improved Question: '{improved_question}'")

def cosine_distance(sentence1: str, sentence2: str, print_result: bool = True) -> tuple[str, float]:
    """Calculate cosine similarity between two sentences using embeddings."""
    from langchain_openai import OpenAIEmbeddings
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    emb1 = embedding_model.embed_query(sentence1)
    emb2 = embedding_model.embed_query(sentence2)
    
    # Reshape for sklearn
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    
    score = cosine_similarity(emb1, emb2)[0][0]
    

    # return a score, i.e. very low, low, medium, high, very high semantic similarity
    if score < 0.2:
        similarity = "very low"
    elif score < 0.4:
        similarity = "low"
    elif score < 0.6:
        similarity = "medium"
    elif score < 0.8:
        similarity = "high"
    else:
        similarity = "very high"
    
    if print_result:
        print(f"Terms: ('{sentence1}', '{sentence2}'), Similarity: {similarity}, Score: {score:.4f}")
    
    return similarity, score

def main():
    """Main function - run the medical RAG workflow test."""
    # test_medical_workflow()
    # test_question_rewriting()

    cosine_distance("the patient is feeling tired", "the patient reports hypothyroidism")
    cosine_distance("the patient is feeling tired", "| TSH (µIU/mL)               | 8.2     | 0.4 – 4.5            | **H** |")
    cosine_distance("the patient is feeling tired", "patient reports fatigue")


if __name__ == "__main__":
    main()

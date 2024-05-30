import spacy

# Load SpaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading the model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

faqs = {
    "What is your return policy?": "Our return policy allows you to return items within 30 days of purchase. If you have any questions, feel free to contact our support team.",
    "How can I track my order?": "You can track your order using the tracking number provided in your confirmation email. Visit our website and enter the tracking number in the 'Track Order' section.",
    "Do you offer international shipping?": "Yes, we offer international shipping to select countries. Shipping rates and delivery times vary depending on the destination.",
    "What payment methods do you accept?": "We accept Visa, MasterCard, American Express, and PayPal. We also offer various local payment methods depending on your country.",
    "Can I change my shipping address?": "If your order hasn't been shipped yet, you can change the shipping address by contacting our customer service.",
    "What is the estimated delivery time?": "The estimated delivery time depends on your location and the shipping method chosen. Typically, it ranges from 3 to 10 business days.",
}


general_responses = {
    "greeting": ["Hello! How can I assist you today?", "Hi there! What can I do for you?", "Greetings! How may I help you?"],
    "goodbye": ["Goodbye! Have a great day!", "Bye! Feel free to reach out if you have more questions.", "Take care!"],
    "thank_you": ["You're welcome!", "Happy to help!", "Anytime!"]
}

def get_intent(user_query):
    user_query = user_query.lower()
    if any(greet in user_query for greet in ["hi", "hello", "hey"]):
        return "greeting"
    if any(bye in user_query for bye in ["bye", "goodbye", "see you"]):
        return "goodbye"
    if any(thanks in user_query for thanks in ["thank you", "thanks"]):
        return "thank_you"
    return "faq"

def find_most_relevant_faq(user_query, faqs):
    user_doc = nlp(user_query)
    highest_similarity = 0
    best_match = None

    for question, answer in faqs.items():
        question_doc = nlp(question)
        similarity = user_doc.similarity(question_doc)

        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = (question, answer)

    return best_match

def chatbot():
    print("Welcome to the FAQ chatbot! How can I help you today?")
    while True:
        user_query = input("You: ")

        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        intent = get_intent(user_query)

        if intent == "greeting":
            print("Chatbot: " + random.choice(general_responses["greeting"]))
        elif intent == "goodbye":
            print("Chatbot: " + random.choice(general_responses["goodbye"]))
            break
        elif intent == "thank_you":
            print("Chatbot: " + random.choice(general_responses["thank_you"]))
        else:
            best_match = find_most_relevant_faq(user_query, faqs)

            if best_match:
                print(f"Chatbot: {best_match[1]}")
            else:
                print("Chatbot: I'm sorry, I don't have an answer for that. Can I help you with anything else?")

if __name__ == "__main__":
    import random
    chatbot()


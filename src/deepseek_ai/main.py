from src.deepseek_ai.chat import handle_user_input

def main():
    user_input = input("Enter a URL: ")
    response = handle_user_input(user_input)
    print(response)

if __name__ == "__main__":
    main()
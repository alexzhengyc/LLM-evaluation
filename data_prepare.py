import pandas as pd

def create_csv():
    data = {
        'prompt_text': ['What is the weather today?', 'Who won the match?', 'How are you?', 'What is your favorite food?'],
        'answer_text': ['The weather is sunny today.', 'The match was won by Team A.', 'I am fine, thank you.', 'My favorite food is pizza.'],
        'quality': [0.9, 0.5, 0.8, 0.6],
        # 'fail_task': [0, 0, 0, 0]
    }

    df = pd.DataFrame(data)
    df['quality'] = df['quality'].astype(float)

    # replicate df 25 times
    df = pd.concat([df]*25, ignore_index=True)

    # Saving dataframe to a CSV file
    df.to_csv('sample_dataset.csv', index=False)



if __name__ == "__main__":
    create_csv()
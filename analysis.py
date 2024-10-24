import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud


def load_clustered_data(csv_filename):
    return pd.read_csv(csv_filename)


def group_poems_by_cluster(df):
    return df.groupby('Cluster')['Poem'].apply(list).reset_index()


def find_common_words(poems):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(poems)
    word_counts = X.toarray().sum(axis=0)
    common_words = sorted(
        zip(vectorizer.get_feature_names_out(), word_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    return common_words[:10]


def calculate_sentiment(poems):
    return [TextBlob(poem).sentiment.polarity for poem in poems]


def average_length(poems):
    return sum(len(poem.split()) for poem in poems) / len(poems)


def plot_wordcloud(words, cluster_number):
    wordcloud = WordCloud(
        width=800, height=400, background_color='white'
    ).generate_from_frequencies(dict(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {cluster_number}')
    plt.show()


def analyze_clusters(df):
    clusters = group_poems_by_cluster(df)

    clusters['Common_Words'] = clusters['Poem'].apply(find_common_words)
    clusters['Sentiment'] = clusters['Poem'].apply(calculate_sentiment)
    clusters['Average_Sentiment'] = clusters['Sentiment'].apply(
        lambda x: sum(x) / len(x)
    )
    clusters['Average_Length'] = clusters['Poem'].apply(average_length)

    for index, row in clusters.iterrows():
        print(f"Cluster {row['Cluster']}:")
        print(f"  Common Words: {row['Common_Words']}")
        print(f"  Average Sentiment: {row['Average_Sentiment']:.2f}")
        print(f"  Average Length: {row['Average_Length']:.2f} words")

        plot_wordcloud(row['Common_Words'], row['Cluster'])


def main():
    csv_filename = 'poems_with_clusters.csv'
    df = load_clustered_data(csv_filename)
    analyze_clusters(df)


if __name__ == "__main__":
    main()

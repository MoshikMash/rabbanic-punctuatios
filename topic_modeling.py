import collections
import itertools
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import umap
from blingfire import text_to_sentences
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
import hdbscan
from sentence_transformers.util import pytorch_cos_sim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder

from LexRank import degree_centrality_scores
import spacy
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

from data_reading import read_df

import nltk

nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def topic_modeling(config):
    if not config['evaluation_mode']:
        # Split data to train and test
        if config['topic_modeling_flags']['split_data_flag']:
            # Load the data
            df = pd.read_csv(config['paths']['all_data_path'])

            if config['only_Mech_Engr_and_Trans_records']:
                df = df[df['Parent_Category'].isin(['Mech_Engr', 'Trans'])]

            # Split the data into train and test sets
            df_train, df_test = train_test_split(df, test_size=config['test_size'])

            # Save the train and test sets to separate CSV files
            path = config['paths']['data_topping_modeling_train_set_path']
            df_train.to_csv(path, index=False)
            print(f'{path} was saved')
            path = config['paths']['data_topping_modeling_test_set_path']
            df_test.to_csv(path, index=False)
            print(f'{path} was saved')

        if config['topic_modeling_flags']['embeddings_sbert_flag']:
            # Split abstracts to sentences
            df_sentences = pd.DataFrame(split_abstracts_to_sentences(config),
                                        columns=['sentence', 'application_number'])
            df_sentences.drop_duplicates(subset='sentence', inplace=True)
            path = config['paths']['df_sentences_path']
            df_sentences.to_csv(path)
            print(f'{path} was saved')

            # Sentence Representation
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            embeddings_sbert = model.encode(df_sentences['sentence'].values, show_progress_bar=True)
            sbert_embeddings_count, sbert_embeddings_size = embeddings_sbert.shape
            print(f'**Sentence-BERT embeddings**: {sbert_embeddings_count} vectors of size {sbert_embeddings_size}.')

            path = config['paths']['embeddings_sbert_path']
            with open(path, 'wb') as file:
                pickle.dump(embeddings_sbert, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{path} was saved')
        else:
            with open(config['paths']['embeddings_sbert_path'], 'rb') as handle:
                embeddings_sbert = pickle.load(handle)
            df_sentences = pd.read_csv(config['paths']['df_sentences_path'])

        if config['topic_modeling_flags']['umap_flag']:
            # UMAP
            embeddings_reduced = umap.UMAP(n_neighbors=config['umap']['n_neighbors'],
                                           n_components=config['umap']['n_components'],
                                           metric=config['umap']['metric']).fit_transform(embeddings_sbert)
            path = config['paths']['embeddings_reduced_path']
            with open(path, 'wb') as file:
                pickle.dump(embeddings_reduced, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{path} was saved')

        else:
            with open(config['paths']['embeddings_reduced_path'], 'rb') as handle:
                embeddings_reduced = pickle.load(handle)

        if config['topic_modeling_flags']['clustering_flag']:
            # clustering
            cluster = hdbscan.HDBSCAN(
                min_cluster_size=config['clustering']['min_cluster_size'],
                metric=config['clustering']['metric'],
                cluster_selection_method=config['clustering']['cluster_selection_method']).fit(embeddings_reduced)
            df_sentences['cluster_id'] = cluster.labels_
            path = config['paths']['df_sentences_sbert_with_clusters_path']
            df_sentences.to_csv(path)
            print(f'{path} was saved')
        else:
            df_sentences = pd.read_csv(config['paths']['df_sentences_sbert_with_clusters_path'])

        # Topic Summarization
        if config['topic_modeling_flags']['topic_summarization']:
            cluster_id_to_central_sentences = {}
            cluster_iter = tqdm(df_sentences.groupby("cluster_id"), desc="Clusters")
            for cluster_id, cluster_group in cluster_iter:
                if cluster_id == -1:
                    continue
                cluster_emb = np.take(embeddings_sbert, cluster_group.index.values, axis=0)
                cos_scores = pytorch_cos_sim(cluster_emb, cluster_emb).numpy()

                centrality_scores = degree_centrality_scores(cos_scores, threshold=None)
                cluster_examples = cluster_group.iloc[
                    np.argsort(-centrality_scores)[:config['topic_summarization']['number_of_centroids_per_cluster']]]
                cluster_id_to_central_sentences[cluster_id] = cluster_examples.sentence.values.tolist()

            df_topics = df_sentences.groupby(['cluster_id'], as_index=False).agg({'sentence': ' '.join}).rename(
                columns={'sentence': 'cluster_sentences'})
            df_topics["central_sentences"] = df_topics["cluster_id"].apply(cluster_id_to_central_sentences.get)
            df_topics["noun_phrases"] = [
                [np for np, _ in
                 get_noun_phrases(examples).most_common(config['topic_summarization']['number_of_nouns_per_cluster'])]
                for examples in tqdm(df_topics["central_sentences"], desc="Processing noun phrases")
            ]

            path = config['paths']['df_topics']
            df_topics.to_csv(path)
            print(f'{path} was saved')
        else:
            df_topics = pd.read_csv(config['paths']['df_topics'])

        if config['topic_modeling_flags']['find_topics']:
            # CREATE TOPIC FEATURES
            lemmatizer = WordNetLemmatizer()
            all_phrases = []
            for phrases in df_topics['noun_phrases'].values:
                if not phrases:
                    continue
                phrases = phrases[1:-1].split(', ')
                for phrase in phrases:
                    # Tokenize the phrase
                    phrase = phrase.lower()
                    tokens = word_tokenize(phrase)
                    # Lemmatize each token in the phrase
                    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
                    # Join the lemmatized tokens back into a phrase
                    lemmatized_phrase = ' '.join(lemmatized_tokens)
                    all_phrases.append(lemmatized_phrase)

            all_phrases = set(all_phrases)
            path = config['paths']['topic_modeling_all_phrases_path']
            with open(path, 'wb') as file:
                pickle.dump(all_phrases, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{path} was saved')
        else:
            with open(config['paths']['topic_modeling_all_phrases_path'], 'rb') as handle:
                all_phrases = pickle.load(handle)

        if config['topic_modeling_flags']['create_topic_features']:
            # Create a CountVectorizer object
            vectorizer = CountVectorizer(vocabulary=all_phrases, binary=True)
            # Fit the CountVectorizer on phrases
            vectorizer.fit(all_phrases)

            df = read_df(config, False)
            lemmatizer = WordNetLemmatizer()
            df['lemmatized_clean_abstract'] = df['clean_abstract'].apply(
                lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(x)]))

            # Transform the abstract column into a sparse matrix of token counts
            one_hot_encoded = vectorizer.transform(df['lemmatized_clean_abstract'])

            # Convert the sparse matrix into a DataFrame
            one_hot_df = pd.DataFrame(one_hot_encoded.toarray(),
                                      columns=[feature for feature in vectorizer.get_feature_names_out()])

            # Concatenate the one-hot encoded DataFrame with the original DataFrame
            df = df.reset_index(drop=True)
            one_hot_df = one_hot_df.reset_index(drop=True)
            result_df = pd.concat([df, one_hot_df], axis=1)
            feature_vector = list(one_hot_df.columns)
            feature_vector.extend(
                [
                    'application_number', 'clean_abstract', 'lemmatized_clean_abstract', 'word_count', 'char_count',
                    'median_word_length', 'avg_word_length', 'skew_word_length',
                    'characters_per_word', 'syll_per_word', 'words_per_sentence', 'sentences_per_paragraph',
                    'type_token_ratio',
                    'syllables', 'sentences', 'long_words', 'complex_words', 'one_if_male', 'Parent_Category',
                    'uspc_class', 'one_if_patented'
                ]
            )
            path = config['paths']['df_with_topic_features_path']
            result_df.to_csv(path)
            print(f'{path} was saved')
            feature_vector = [feature for feature in feature_vector if
                              feature not in ['application_number', 'clean_abstract', 'lemmatized_clean_abstract']]
            feature_vector.remove('')
            path = config['paths']['topic_modeling_feature_vector_path']
            with open(path, 'wb') as file:
                pickle.dump(feature_vector, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{path} was saved')
        else:
            result_df = pd.read_csv(config['paths']['df_with_topic_features_path'])
            with open(config['paths']['topic_modeling_feature_vector_path'], 'rb') as handle:
                feature_vector = pickle.load(handle)

        if config['topic_modeling_flags']['training']:
            # Train a random forest classifier
            rf_classifier = RandomForestClassifier(
                n_estimators=config['topic_modeling_training_parameters']['n_estimators'],
                max_depth=config['topic_modeling_training_parameters']['max_depth'],
                random_state=config['random_state'])
            result_df = result_df[feature_vector]

            # Encode categorical variables
            le_category = LabelEncoder()
            result_df['Parent_Category'] = le_category.fit_transform(result_df['Parent_Category'])
            le_uspc_class = LabelEncoder()
            result_df['uspc_class'] = le_uspc_class.fit_transform(result_df['uspc_class'].astype(str))
            le_patent = LabelEncoder()
            result_df['one_if_patented'] = le_patent.fit_transform(result_df['one_if_patented'])

            if config['topic_summarization']['features_to_drop']:
                feature_vector = [x for x in feature_vector if
                                  x not in config['topic_summarization']['features_to_drop']]
            feature_vector.remove('one_if_patented')

            # Under-sample the target variable
            rus = RandomUnderSampler(random_state=config['random_state'])
            X_resampled, y_resampled = rus.fit_resample(result_df[feature_vector], result_df['one_if_patented'])

            # Fit the model
            rf_classifier.fit(X_resampled, y_resampled)

            # Save the trained model
            path = config['paths']['topic_modeling_rf_classifier_path']
            with open(path, 'wb') as file:
                pickle.dump(rf_classifier, file, protocol=pickle.HIGHEST_PROTOCOL)
                print(f'{path} was saved')

            # Evaluate the model using the training set
            y_pred = rf_classifier.predict(result_df[feature_vector])
            print(classification_report(result_df['one_if_patented'], y_pred))

            # Get feature importances
            feature_importances = rf_classifier.feature_importances_

            # Create a DataFrame to save the feature importances
            importance_df = pd.DataFrame({
                'Feature': feature_vector,
                'Importance': feature_importances
            })

            # Sort the DataFrame by importance (optional)
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            # Save to a CSV file
            results_folder_path = pathlib.Path(config['paths']['results_folder_path']) / 'topic_modeling'
            # Create the directory if it does not exist
            os.makedirs(results_folder_path, exist_ok=True)
            importance_df.to_csv(results_folder_path / 'feature_importances.csv', index=False)

            print(f'under sampled date length: {len(y_resampled)}')
            # Confusion Matrix
            cm = confusion_matrix(result_df['one_if_patented'], y_pred)
            # Since the target is binary (0 or 1), define the class names
            class_names = ['not patented', 'patented']  # Adjust these names as needed

            # Convert the confusion matrix to a DataFrame for better readability
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

            # Visualize the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            plt.show()
    if config['topic_modeling_flags']['grid_search']:
        # Define the parameter grid for grid search
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [50, 100, 200, 300, 400]
        }

        result_df = result_df[feature_vector]

        # Encode categorical variables
        le_category = LabelEncoder()
        result_df['Parent_Category'] = le_category.fit_transform(result_df['Parent_Category'])
        le_uspc_class = LabelEncoder()
        result_df['uspc_class'] = le_uspc_class.fit_transform(result_df['uspc_class'].astype(str))
        le_patent = LabelEncoder()
        result_df['one_if_patented'] = le_patent.fit_transform(result_df['one_if_patented'])

        if config['topic_summarization']['features_to_drop']:
            feature_vector = [x for x in feature_vector if x not in config['topic_summarization']['features_to_drop']]
        feature_vector.remove('one_if_patented')

        # Set up KFold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=config['random_state'])

        # Initialize the random forest classifier
        rf_classifier = RandomForestClassifier(random_state=config['random_state'])

        # Perform grid search with cross-validation
        grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=kf, n_jobs=-1, scoring='accuracy',
                                   verbose=3)

        # Under-sample the target variable
        rus = RandomUnderSampler(random_state=config['random_state'])
        X_resampled, y_resampled = rus.fit_resample(result_df[feature_vector], result_df['one_if_patented'])

        # Fit the grid search
        grid_search.fit(X_resampled, y_resampled)

        # Get the best parameters and model
        best_params = grid_search.best_params_
        best_rf_classifier = grid_search.best_estimator_

        # Save the best model
        path = config['paths']['topic_modeling_rf_classifier_path']
        with open(path, 'wb') as file:
            pickle.dump(best_rf_classifier, file, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'{path} was saved')

        # Evaluate the model using the entire dataset
        y_pred = best_rf_classifier.predict(result_df[feature_vector])
        print(classification_report(result_df['one_if_patented'], y_pred))

        # Get feature importances
        feature_importances = best_rf_classifier.feature_importances_

        # Create a DataFrame to save the feature importances
        importance_df = pd.DataFrame({
            'Feature': feature_vector,
            'Importance': feature_importances
        })

        # Sort the DataFrame by importance (optional)
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Save to a CSV file
        results_folder_path = pathlib.Path(config['paths']['results_folder_path']) / 'topic_modeling'
        os.makedirs(results_folder_path, exist_ok=True)
        importance_df.to_csv(results_folder_path / 'feature_importances.csv', index=False)

        print(f'Best parameters found: {best_params}')
        print(f'Confusion Matrix after cross-validation:')
        cm = confusion_matrix(result_df['one_if_patented'], y_pred)
        print(cm)
    else:
        # Evaluation
        # Load all phrases
        with open(config['paths']['topic_modeling_all_phrases_path'], 'rb') as handle:
            all_phrases = pickle.load(handle)

        # Create a CountVectorizer object
        vectorizer = CountVectorizer(vocabulary=all_phrases, binary=True)

        # Fit the CountVectorizer on phrases
        vectorizer.fit(all_phrases)

        # Load the test data
        df = pd.read_csv(config['paths']['data_topping_modeling_test_data_path'])

        # Lemmatize the clean abstracts
        lemmatizer = WordNetLemmatizer()
        df['lemmatized_clean_abstract'] = df['clean_abstract'].apply(
            lambda x: ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(x)])
        )

        # Transform the abstract column into a sparse matrix of token counts
        one_hot_encoded = vectorizer.transform(df['lemmatized_clean_abstract'])

        # Convert the sparse matrix into a DataFrame
        one_hot_df = pd.DataFrame(one_hot_encoded.toarray(),
                                  columns=[feature for feature in vectorizer.get_feature_names_out()])

        # Concatenate the one-hot encoded DataFrame with the original DataFrame
        df = df.reset_index(drop=True)
        one_hot_df = one_hot_df.reset_index(drop=True)
        test_df = pd.concat([df, one_hot_df], axis=1)

        # Encode categorical features
        le_category = LabelEncoder()
        test_df['Parent_Category'] = le_category.fit_transform(test_df['Parent_Category'])
        le_uspc_class = LabelEncoder()
        test_df['uspc_class'] = le_uspc_class.fit_transform(test_df['uspc_class'].astype(str))
        le_patent = LabelEncoder()
        test_df['one_if_patented'] = le_patent.fit_transform(test_df['one_if_patented'])

        # Load the classifier and feature vector
        with open(config['paths']['topic_modeling_rf_classifier_path'], 'rb') as handle:
            rf_classifier = pickle.load(handle)
        with open(config['paths']['topic_modeling_feature_vector_path'], 'rb') as handle:
            feature_vector = pickle.load(handle)

        # Remove specified features
        if config['topic_summarization']['features_to_drop']:
            feature_vector = [x for x in feature_vector if x not in config['topic_summarization']['features_to_drop']]
        feature_vector.remove('one_if_patented')

        # Apply undersampling
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(test_df[feature_vector], test_df['one_if_patented'])

        # Make predictions on the resampled test set
        y_pred = rf_classifier.predict(X_resampled)

        # Evaluate the performance of the model
        print(classification_report(y_resampled, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_resampled, y_pred)

        # Define class names
        class_names = ['not patented', 'patented']  # Adjust these names as needed

        # Convert the confusion matrix to a DataFrame for better readability
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        # Visualize the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()


def get_noun_phrases(corpus):
    nlp = spacy.load("en_core_web_sm")
    res = collections.Counter()
    if not corpus:
        return res
    for doc in nlp.pipe(corpus):
        noun_pharses = set()
        for nc in doc.noun_chunks:
            for np in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]:
                noun_pharses.add(np)
                if np[0].tag_ in ("PRP", "WP", "NNP"):
                    noun_pharses.add(np)
            res[nc] += 1
    return res


def split_abstracts_to_sentences(config):
    df = read_df(config, False)
    sentence_items = []
    for index, row in df.iterrows():
        abstract_sentences = text_to_sentences(row['abstract']).split("\n")
        sentence_with_idx = itertools.zip_longest(abstract_sentences, [row['application_number']],
                                                  fillvalue=row['application_number'])
        sentence_items.extend(sentence_with_idx)

    return sentence_items

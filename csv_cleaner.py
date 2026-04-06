import pandas as pd

# Load your YouTube comment CSV
df = pd.read_csv('yt_comments.csv')

# List the columns you WANT to get rid of
to_remove = ['channel_username' , 'channel_id', 'video_id', 'video_title', 'comment_id', 'author', 'published_at', 'like_count', 'reply_count']

# Drop them and save
df_cleaned = df.drop(columns=to_remove)
df_cleaned.to_csv('cleaned_comments.csv', index=False)
from googleapiclient.discovery import build
import time
import pandas as pd

def main():

    start_time = time.time()
    youtube = Youtube().youtube
    channel_id = "UCX6OQ3DkcsbYNE6H8uQQuVA" # please write channel ID
    channel = Channel(channel_id, youtube)
    videoCount = int(channel.channel_info["statistics"]["videoCount"])
    Data = pd.DataFrame(channel.GetVideoData(youtube,videoCount))
    Data.to_csv(f"{channel.channel_data[0]}_metadata.csv")
    end_time = time.time()
    print(end_time - start_time)


class Youtube:
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    API_KEY = "AIzaSyDlXZQ4Ymm31hHViYWCdBC2cSGhp_uUVbY" # Please write your API key.
    youtube = build(
    YOUTUBE_API_SERVICE_NAME,
    YOUTUBE_API_VERSION,
    developerKey=API_KEY
    )



class Channel:
    def __init__(self, channel_id,youtube):
        self.channel_id = channel_id
        self.channel_info = youtube.channels().list(part="contentDetails,snippet,statistics", id=channel_id).execute()["items"][0]
        channel_title = self.channel_info["snippet"]["title"]
        subscriber_count = self.channel_info["statistics"].get("subscriberCount")
        channel_age = self.channel_info["snippet"]["publishedAt"]
        total_uploads = self.channel_info["statistics"]["videoCount"]
        self.channel_data = [channel_title, subscriber_count, channel_age, total_uploads]

    def GetVideoID(self,youtube,video_number):
        self.playlist_id = self.channel_info["contentDetails"]["relatedPlaylists"]["uploads"]
        metadata = []
        self.video_ids = []
        nextPageToken = None
        while int(video_number) > 50:
            response = youtube.playlistItems().list(
            part="contentDetails",
            playlistId=self.playlist_id,
            maxResults=50,
            pageToken=nextPageToken
            ).execute()
            nextPageToken = response.get("nextPageToken")
            video_number -= 50
            self.video_ids += [it["contentDetails"]["videoId"]
                  for it in response["items"]]
        response = youtube.playlistItems().list(
        part="contentDetails",
        playlistId=self.playlist_id,
        maxResults=video_number,
        pageToken=nextPageToken
        ).execute()
        self.video_ids += [it["contentDetails"]["videoId"]
                  for it in response["items"]]




    def GetVideoData(self,youtube,video_number):
        self.GetVideoID(youtube,video_number)
        metadata = []
        index = 0
        while index < len(self.video_ids):
            videos = youtube.videos().list(
                    part = "snippet,statistics,contentDetails",
                    id = ",".join(self.video_ids[index:index + 50])
                    ).execute()
            index += 50
            for video in videos["items"]:
                title = video["snippet"]["title"]
                thumbnail = video["snippet"]["thumbnails"]["default"]["url"]
                desc_len = len(video["snippet"]["description"])
                tags = video["snippet"].get("tags", [])
                published = video["snippet"]["publishedAt"]
                view_count = video["statistics"].get("viewCount",0)
                like_count = video["statistics"].get("likeCount")
                comment_count = video["statistics"].get("commentCount")
                duration = video["contentDetails"]["duration"].replace('PT','').replace('H','時間').replace('M','分').replace('S','秒')
                category = video["snippet"]["categoryId"]
                video_data = [
                    title, thumbnail,
                    desc_len, tags, published,
                    view_count, like_count,
                    comment_count, duration,
                    category
                    ]
                metadata.append(video_data + self.channel_data)
        sorted_metadata = pd.DataFrame(sorted(metadata, key = lambda x: int(x[5]) or 0, reverse=True))
        sorted_metadata.columns = ["Video Title", "Thumbnail URL", "Description Length",
                            "Tags", "Published Data & Time", "View Count", "Like Count",
                            "Comment Count", "Duration", "Category", "Channel Title",
                            "Channel Subscriber", "Channel Age", "Total number of channel uploads"]
        return sorted_metadata

if __name__ == "__main__":
    main()

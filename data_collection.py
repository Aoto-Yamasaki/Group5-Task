from googleapiclient.discovery import build
import time
import pandas as pd
import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()


def main():

    start_time = time.time()
    youtube = Youtube().youtube
    channel_id = os.getenv("CHANNEL_ID")
    if not channel_id:
        raise ValueError("CHANNEL_ID is not set in the .env file")
    channel = Channel(channel_id, youtube)
    videoCount = channel.channel_info["statistics"]["videoCount"]
    Data = pd.DataFrame(channel.GetVideoData(youtube, 5))
    Data.to_csv(f"{channel.channel_data[0]}_metadata.csv", index=False)
    end_time = time.time()
    print(end_time - start_time)


class Youtube:
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    API_KEY = os.getenv("API_KEY")
    if not API_KEY:
        raise ValueError("API_KEY is not set in the .env file")
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)


class Channel:
    def __init__(self, channel_id, youtube):
        self.channel_id = channel_id
        self.channel_info = (
            youtube.channels()
            .list(part="contentDetails,snippet,statistics", id=channel_id)
            .execute()["items"][0]
        )
        channel_title = self.channel_info["snippet"]["title"]
        subscriber_count = self.channel_info["statistics"].get("subscriberCount")
        channel_age = self.channel_info["snippet"]["publishedAt"]
        total_uploads = self.channel_info["statistics"]["videoCount"]
        self.channel_data = [
            channel_title,
            subscriber_count,
            channel_age,
            total_uploads,
        ]

    def GetVideoID(self, youtube, video_number):
        self.playlist_id = self.channel_info["contentDetails"]["relatedPlaylists"][
            "uploads"
        ]
        metadata = []
        self.video_ids = []
        nextPageToken = None
        while video_number > 50:
            response = (
                youtube.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=self.playlist_id,
                    maxResults=50,
                    pageToken=nextPageToken,
                )
                .execute()
            )
            nextPageToken = response.get("nextPageToken")
            video_number -= 50
            self.video_ids += [
                it["contentDetails"]["videoId"] for it in response["items"]
            ]
        response = (
            youtube.playlistItems()
            .list(
                part="contentDetails",
                playlistId=self.playlist_id,
                maxResults=video_number,
                pageToken=nextPageToken,
            )
            .execute()
        )
        self.video_ids += [it["contentDetails"]["videoId"] for it in response["items"]]

    def GetVideoData(self, youtube, video_number):
        self.GetVideoID(youtube, video_number)
        # DataFrameのカラム名を定義
        columns = [
            "Video Title",
            "Thumbnail URL",
            "Description Length",
            "Tags",
            "Published Data & Time",
            "View Count",
            "Like Count",
            "Comment Count",
            "Duration",
            "Category",
            "Channel Title",
            "Channel Subscriber",
            "Channel Age",
            "Total number of channel uploads",
        ]
        # 空のDataFrameを初期化
        metadata = pd.DataFrame(columns=columns)

        index = 0
        while index < len(self.video_ids):
            videos = (
                youtube.videos()
                .list(
                    part="snippet,statistics,contentDetails",
                    id=",".join(self.video_ids[index : index + 50]),
                )
                .execute()
            )
            index += 50
            for video in videos["items"]:
                title = video["snippet"]["title"]
                thumbnail = video["snippet"]["thumbnails"]["default"]["url"]
                desc_len = len(video["snippet"]["description"])
                tags = video["snippet"].get("tags", [])
                published = video["snippet"]["publishedAt"]
                view_count = int(video["statistics"].get("viewCount", 0))
                like_count = video["statistics"].get("likeCount")
                comment_count = video["statistics"].get("commentCount")
                duration = (
                    video["contentDetails"]["duration"]
                    .replace("PT", "")
                    .replace("H", "時間")
                    .replace("M", "分")
                    .replace("S", "秒")
                )
                category = video["snippet"]["categoryId"]

                # 新しい行データを作成
                new_row = pd.DataFrame(
                    {
                        "Video Title": [title],
                        "Thumbnail URL": [thumbnail],
                        "Description Length": [desc_len],
                        "Tags": [tags],
                        "Published Data & Time": [published],
                        "View Count": [view_count],
                        "Like Count": [like_count],
                        "Comment Count": [comment_count],
                        "Duration": [duration],
                        "Category": [category],
                        "Channel Title": [self.channel_data[0]],
                        "Channel Subscriber": [self.channel_data[1]],
                        "Channel Age": [self.channel_data[2]],
                        "Total number of channel uploads": [self.channel_data[3]],
                    }
                )

                # DataFrameに行を追加
                metadata = pd.concat([metadata, new_row], ignore_index=True)
            # while True:
            #     playlist_items = youtube.playlistItems().list(
            #     part="contentDetails",
            #     playlistId=self.playlist_id,
            #     maxResults=50,
            #     pageToken=nextPageToken
            #     ).execute()
            #     for item in playlist_items["items"]:
            #         video_id = item["contentDetails"]["videoId"]
            #         video = youtube.videos().list(
            #             part = "statistics",
            #             id = video_id
            #             ).execute()
            #         self.videolist_id.append([video_id,video])
        # View Countでソートして返す
        sorted_metadata = metadata.sort_values(by="View Count", ascending=False)
        return sorted_metadata


if __name__ == "__main__":
    main()

# Images

import json
import httpx
import os
import time

# Set up the HTTP client with appropriate headers
client = httpx.Client(
    headers={
        "x-ig-app-id": "936619743392459",  # Instagram's internal app ID
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
    }
)


# Function to scrape posts from Instagram using the user ID
def scrape_user_posts(user_id: str, max_pages: int = None):
    base_url = "https://www.instagram.com/graphql/query/?query_hash=e769aa130647d2354c40ea6a439bfc08&variables="
    variables = {
        "id": user_id,
        "first": 12,  # Number of posts to retrieve per page
        "after": None,
    }
    all_posts = []
    page_count = 0

    while True:
        try:
            result = client.get(base_url + json.dumps(variables))
            data = result.json()
            posts = data["data"]["user"]["edge_owner_to_timeline_media"]
            all_posts.extend(posts["edges"])

            # Pagination handling
            page_info = posts["page_info"]
            if not page_info["has_next_page"]:
                break

            variables["after"] = page_info["end_cursor"]
            page_count += 1
            print(f"Page {page_count}: Retrieved {len(posts['edges'])} posts.")

            # Add a delay to avoid hitting rate limits
            time.sleep(2)  # Wait 2 seconds before the next request

            # Stop if max_pages limit is reached
            if max_pages and page_count >= max_pages:
                break

        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 60 seconds...")
            time.sleep(60)  # Wait 60 seconds before retrying the request

    return all_posts


# Function to download images and save them to Seagate Backup Plus Drive
def download_images(posts, save_path):
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for post in posts:
        image_url = post["node"]["display_url"]
        image_id = post["node"]["id"]
        img_data = client.get(image_url).content

        # Save the image to the specified path
        with open(f"{save_path}/{image_id}.jpg", 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded image {image_id}")


# Main script
if __name__ == "__main__":
    # Define the user ID for Company
    user_id = "3133062890"

    # Scrape posts with proper pagination and delay (set max_pages to None to get all pages)
    peta_posts = scrape_user_posts(user_id, max_pages=None)

    # Specify the path to your Seagate Backup Plus Drive
    save_directory = "/Volumes/Seagate Backup Plus Drive/Impossible_Foods_Instagram_Images"

    # Download images to the Seagate Backup Plus Drive
    download_images(peta_posts, save_directory)

# Images Part 2

import json
import httpx
import os
import time

# Set up the HTTP client with appropriate headers
client = httpx.Client(
    headers={
        "x-ig-app-id": "936619743392459",  # Instagram's internal app ID
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "*/*",
    }
)

# Function to scrape posts from Instagram using the user ID
def scrape_user_posts(user_id: str, max_pages: int = None):
    base_url = "https://www.instagram.com/graphql/query/?query_hash=e769aa130647d2354c40ea6a439bfc08&variables="
    variables = {
        "id": user_id,
        "first": 12,  # Number of posts to retrieve per page
        "after": None,
    }
    all_posts = []
    page_count = 0
    retry_count = 0  # Tracks the number of retries after an error
    max_retries = 2  # Maximum number of retries before waiting for 5 minutes
    while True:
        try:
            result = client.get(base_url + json.dumps(variables))
            result.raise_for_status()  # Raise an HTTP error if one occurred
            data = result.json()
            posts = data["data"]["user"]["edge_owner_to_timeline_media"]
            all_posts.extend(posts["edges"])

            # Pagination handling
            page_info = posts["page_info"]
            if not page_info["has_next_page"]:
                break

            variables["after"] = page_info["end_cursor"]
            page_count += 1
            print(f"Page {page_count}: Retrieved {len(posts['edges'])} posts.")

            # Add a delay to avoid hitting rate limits
            time.sleep(2)  # Wait 2 seconds before the next request

            # Stop if max_pages limit is reached
            if max_pages and page_count >= max_pages:
                break

            # Reset the retry count after a successful request
            retry_count = 0

        except Exception as e:
            print(f"An error occurred: {e}. Retrying in 60 seconds...")
            retry_count += 1

            # If the error has occurred more than max_retries times, wait for 5 minutes
            if retry_count > max_retries:
                print(f"Error occurred {retry_count} times. Waiting for 5 minutes before retrying...")
                time.sleep(300)  # Wait 5 minutes
                retry_count = 0  # Reset retry count after waiting

            else:
                # Wait 60 seconds before retrying
                time.sleep(60)

            # If the same error occurs again after waiting, stop and download what we have
            if retry_count == max_retries:
                print(f"Error persists. Downloading what we have so far...")
                break

    return all_posts

# Function to download images and save them to the specified path
def download_images(posts, save_path):
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for post in posts:
        image_url = post["node"]["display_url"]
        image_id = post["node"]["id"]
        img_data = client.get(image_url).content

        # Save the image to the specified path
        with open(f"{save_path}/{image_id}.jpg", 'wb') as handler:
            handler.write(img_data)
        print(f"Downloaded image {image_id}")

# Main script
if __name__ == "__main__":
    # Define the user ID for the Instagram account
    user_id = "3133062890"

    # Scrape posts with proper pagination and delay (set max_pages to None to get all pages)
    retrieved_posts = scrape_user_posts(user_id, max_pages=None)

    # Specify the path to your Seagate Backup Plus Drive
    save_directory = "/Volumes/Seagate Backup Plus Drive/Impossible_Foods_Instagram_Images"

    # Download images to the Seagate Backup Plus Drive
    download_images(retrieved_posts, save_directory)

# Images Part 3:

import instaloader
import os
import time

# Function to scrape posts using Instaloader with authentication
def scrape_user_posts_instaloader(username, max_posts=None):
    L = instaloader.Instaloader()

    # Login with your Instagram credentials (optional but recommended)
    # Replace 'your_username' and 'your_password' with actual credentials
    L.login('jamesjonah556', 'Rocker12')

    # Load the profile by username
    profile = instaloader.Profile.from_username(L.context, username)

    all_posts = []
    post_count = 0

    # Loop through the user's posts
    for post in profile.get_posts():
        all_posts.append(post)
        post_count += 1
        print(f"Post {post_count}: {post.date} - {post.url}")

        # Stop if max_posts limit is reached
        if max_posts and post_count >= max_posts:
            break

    return all_posts

# Function to download images from scraped posts
def download_images(posts, save_path):
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for post in posts:
        image_url = post.url
        image_id = post.mediaid

        # Use Instaloader's download method to download images
        L = instaloader.Instaloader()
        try:
            L.download_post(post, target=f"{save_path}/{image_id}")
            print(f"Downloaded image from post {image_id}")
        except Exception as e:
            print(f"Error downloading image {image_id}: {e}")

# Main script
if __name__ == "__main__":
    # Define the username for the Instagram account (replace 'your_username' with the actual username)
    username = "impossible_foods"

    # Scrape posts with a limit on the number of posts (set max_posts to None to get all posts)
    posts = scrape_user_posts_instaloader(username, max_posts=None)

    # Specify the path to your Seagate Backup Plus Drive
    save_directory = "/Volumes/Seagate Backup Plus Drive/Impossible_Foods_Instagram_Images"

    # Download images to the Seagate Backup Plus Drive
    download_images(posts, save_directory)

# Image Part 4:

import instaloader
import os

# Function to scrape posts using Instaloader with authentication
def scrape_user_posts_instaloader(username, max_posts=None):
    L = instaloader.Instaloader(download_comments=True)  # Enable comment downloading

    # Login with your Instagram credentials
    L.login('jamesjonah556', 'Rocker12')

    # Load the profile by username
    profile = instaloader.Profile.from_username(L.context, username)

    all_posts = []
    post_count = 0

    # Loop through the user's posts
    for post in profile.get_posts():
        all_posts.append(post)
        post_count += 1
        print(f"Post {post_count}: {post.date} - {post.url}")

        # Stop if max_posts limit is reached
        if max_posts and post_count >= max_posts:
            break

    return all_posts

# Function to download posts and metadata, including comments
def download_images(posts, save_path):
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Use Instaloader's download_post method to download the post with comments
    L = instaloader.Instaloader(download_comments=True)  # Ensure comments are downloaded

    for post in posts:
        try:
            # Downloads the post along with metadata and comments
            L.download_post(post, target=save_path)
            print(f"Downloaded post with comments: {post.mediaid}")
        except Exception as e:
            print(f"Error downloading post {post.mediaid}: {e}")

# Main script
if __name__ == "__main__":
    # Define the username for the Instagram account
    username = "wholefoods"

    # Scrape posts with a limit on the number of posts (set max_posts to None to get all posts)
    posts = scrape_user_posts_instaloader(username, max_posts=None)

    # Specify the path to your Seagate Backup Plus Drive (corrected path)
    save_directory = "/Volumes/Seagate Backup Plus Drive/WholeFoods_Instagram_Images"

    # Download posts and metadata (including comments) to the Seagate Backup Plus Drive
    download_images(posts, save_directory)

# Part 5: Images + No repeats

import instaloader
import os


# Function to scrape posts using Instaloader with authentication
def scrape_user_posts_instaloader(username, max_posts=None):
    L = instaloader.Instaloader(download_comments=True)  # Enable comment downloading

    # Login with your Instagram credentials
    L.login('jamesjonah556', 'Rocker12')

    # Load the profile by username
    profile = instaloader.Profile.from_username(L.context, username)

    all_posts = []
    post_count = 0

    # Loop through the user's posts
    for post in profile.get_posts():
        all_posts.append(post)
        post_count += 1
        print(f"Post {post_count}: {post.date} - {post.url}")

        # Stop if max_posts limit is reached
        if max_posts and post_count >= max_posts:
            break

    return all_posts


# Function to download posts and metadata, including comments
def download_images(posts, save_path):
    # Ensure the save directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Use Instaloader's download_post method to download the post with comments
    L = instaloader.Instaloader(download_comments=True)  # Ensure comments are downloaded

    for post in posts:
        # Define the post directory based on the post ID
        post_directory = os.path.join(save_path, str(post.mediaid))

        # Check if the directory for the post already exists
        if os.path.exists(post_directory):
            print(f"Post {post.mediaid} already downloaded. Skipping...")
            continue  # Skip downloading if the post already exists

        try:
            # Downloads the post along with metadata and comments
            L.download_post(post, target=save_path)
            print(f"Downloaded post with comments: {post.mediaid}")
        except Exception as e:
            print(f"Error downloading post {post.mediaid}: {e}")


# Main script
if __name__ == "__main__":
    # Define the username for the Instagram account
    username = "wholefoods"

    # Scrape posts with a limit on the number of posts (set max_posts to None to get all posts)
    posts = scrape_user_posts_instaloader(username, max_posts=None)

    # Specify the path to your Seagate Backup Plus Drive (corrected path)
    save_directory = "/Volumes/Seagate Backup Plus Drive/WholeFoods_Instagram_Images"

    # Download posts and metadata (including comments) to the Seagate Backup Plus Drive
    download_images(posts, save_directory)

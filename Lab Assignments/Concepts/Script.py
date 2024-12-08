import instaloader

def get_followers_followings(username):
    # Create an instance of Instaloader
    L = instaloader.Instaloader()

    # Login with your Instagram credentials
    username = input("Enter your Instagram username: ")
    password = input("Enter your Instagram password: ")
    L.login(username, password)

    # Load the profile of the logged-in user
    profile = instaloader.Profile.from_username(L.context, username)

    # Get followers and followings
    followers = set([follower.username for follower in profile.get_followers()])
    followings = set([following.username for following in profile.get_followees()])

    return followers, followings

def find_non_followers(followers, followings):
    # Find the people you are following but who don't follow you back
    non_followers = followings - followers
    return non_followers

if __name__ == "__main__":
    # Get followers and followings
    followers, followings = get_followers_followings(input("Enter your Instagram username: "))
    
    # Find non-followers
    non_followers = find_non_followers(followers, followings)
    
    # Print the users who haven't followed you back
    print(f"Users who haven't followed you back: {non_followers}")

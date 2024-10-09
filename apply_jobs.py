from multion.client import MultiOn

client = MultiOn(
    api_key="89ace9e30b124ace852e05af1dd1ef16"
)

browse_response = client.browse(
    cmd="Find the top comment of the top post on Hackernews.",
    url="https://news.ycombinator.com/"
)
print("Browse response:", browse_response)

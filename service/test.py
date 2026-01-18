import requests

# URL of your locally running function (e.g., FastAPI, Flask, or AWS Lambda runtime)
url = "http://localhost:8080/predict"

request = {
    "url": "https://www.fisheries.noaa.gov/s3//styles/original/s3/2022-08/640x427-Shrimp-Atlantic-Northern-NOAAFisheries.png?itok=ekHT_GTn"
}

response = requests.post(url, json=request)
result = response.json()

print(f"Top prediction: {result['top_class']} ({result['top_probability']:.2%})")
print(f"\nAll predictions:")
for cls, prob in result['predictions'].items():
    print(f"  {cls:12s}: {prob:.2%}")
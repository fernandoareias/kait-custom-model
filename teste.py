import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://testehub2096476304.openai.azure.com/openai/deployments/gpt-4o-mini"
model_name = "gpt-4o-mini"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential("7kDDK1KyAH1437oEp2823O7kiEPOS2DwuLbq8BYBaVu5IMNcxL5uJQQJ99BCACZoyfiXJ3w3AAAAACOGWUCx"),
)

response = client.complete(
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="I am going to Paris, what should I see?")
    ],
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    model=model_name
)

print(response.choices[0].message.content)
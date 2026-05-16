import asyncio
import json
import os
from apyhiveapi import Auth

async def get_tokens():
    user = input("Hive Email: ")
    passw = input("Hive Password: ")

    auth = Auth(username=user, password=passw)
    
    print("Logging in...")
    result = await auth.login()

    # If it's an SMS challenge
    if isinstance(result, dict) and result.get("ChallengeName") == "SMS_MFA":
        code = input("Enter SMS Code sent to your phone: ")
        tokens = await auth.sms_2fa(code, result)
    else:
        # If no SMS was needed, the 'result' IS the tokens!
        tokens = result

    # THIS IS THE FIX: Actually save the tokens to a JSON file
    with open("hive_tokens.json", "w") as f:
        json.dump(tokens, f)
        
    print("✅ Tokens successfully saved to hive_tokens.json!")

if __name__ == "__main__":
    asyncio.run(get_tokens())

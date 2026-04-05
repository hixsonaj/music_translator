import asyncio
from uagents.communication import send_sync_message
from models import LyricLine, AcceptedLine

ORCHESTRATOR_ADDRESS = "agent1qga4t2sj75u7kxrw8gpj3jpx0wg5dch0jc06elsfmc54mrjsdntyukzvkgf"

english_to_spanish = [
    ("Never gonna give you up", 7),
    ("Never gonna let you down", 7),
    ("Never gonna run around and desert you", 11),
    ("Never gonna make you cry", 6),
    ("Never gonna say goodbye", 6),
    ("Never gonna tell a lie and hurt you", 10),
]

spanish_to_english = [
    ("Nunca voy a rendirme", 7),
    ("Siempre estaré a tu lado", 7),
    ("Te quiero con todo mi corazón", 9),
    ("Eres la luz de mi vida", 7),
]

async def translate_line(line_id, text, target_lang, syllables):
    result = await send_sync_message(
        destination=ORCHESTRATOR_ADDRESS,
        message=LyricLine(
            line_id=line_id,
            original=text,
            translated="",
            target_lang=target_lang,
            target_syllables=syllables,
        ),
        response_type=AcceptedLine,
        timeout=60,
    )
    return result

async def main():
    print("=" * 50)
    print("English → Spanish")
    print("=" * 50)
    for i, (text, syllables) in enumerate(english_to_spanish):
        result = await translate_line(i, text, "es", syllables)
        print(f"  Original:    {text}")
        print(f"  Translation: {result.final_translation}")
        print(f"  Attempts:    {result.attempts}")
        print()

    print("=" * 50)
    print("Spanish → English")
    print("=" * 50)
    for i, (text, syllables) in enumerate(spanish_to_english):
        result = await translate_line(i + 100, text, "en", syllables)
        print(f"  Original:    {text}")
        print(f"  Translation: {result.final_translation}")
        print(f"  Attempts:    {result.attempts}")
        print()

asyncio.run(main())
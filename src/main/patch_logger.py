from pathlib import Path

path = Path("doall.py")
text = path.read_text(encoding="utf-8")

start = text.find("def ask(")
if start == -1:
    raise RuntimeError("ask() not found")

end = text.find("\ndef ", start + 1)
if end == -1:
    raise RuntimeError("End of ask() not found")

new_ask = """def ask(oracle_id, context, summaries, example_method, test_name, before_code, test_code, focal_code):
    global conversation_history

    log_event(
        "oracle_generation_start",
        oracle_id=oracle_id,
        test_name=test_name,
        test_code=test_code,
        focal_code=focal_code,
        context_history=context.history
    )

    try:
        oracle_prompt_generator(
            oracle_id, context, summaries, example_method,
            before_code, test_code, focal_code
        )

        print(context.history)

        gpt_oracle = get_gpt_oracle(
            test_name=test_name,
            context=context
        )

        status_count['total'] += 1

        log_event(
            "oracle_generation_complete",
            oracle_id=oracle_id,
            test_name=test_name,
            generated_oracle=gpt_oracle,
            context_history=context.history
        )

        return gpt_oracle

    except Exception as e:
        log_event(
            "oracle_generation_error",
            oracle_id=oracle_id,
            test_name=test_name,
            error_type=type(e).__name__,
            error=str(e)
        )
        raise

"""

text = text[:start] + new_ask + text[end + 1:]
path.write_text(text, encoding="utf-8")

print("ASK PATCH SUCCESSFUL")

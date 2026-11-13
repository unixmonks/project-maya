"""serve/test_beam_frontend.py - Reflection Beam-1.0's chat template (tools/mm3/beam_chat_template.jinja) through the
server's ChatTemplate (detection, effort levels, a tool result named after its call), and its output (harmony_xml)
through BeamRewriter + OutputParser: reasoning, answers, typed tool calls - streamed in pieces and whole."""
from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from serve.frontend import ChatTemplate, OutputParser, openai_to_messages  # noqa: E402


class BeamFrontend(unittest.TestCase):
    def test_template_and_parser(self):
        failed = []

        def check(ok, what, detail=""):
            if not ok:
                failed.append(what + "  " + str(detail)[:300])

        tpl = ChatTemplate(ROOT / "tools" / "mm3" / "beam_chat_template.jinja")
        check(tpl.beam and not tpl.qwen and not tpl.minimax, "the template is detected as Beam's")
        tools = [{"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "max_lines": {"type": "integer"}, "opts": {"type": "object"}}, "required": ["path"]}}]
        msgs = [{"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Read /etc/hosts"},
                {"role": "assistant", "content": "", "reasoning_content": "I should read it.",
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "read_file", "arguments": json.dumps({"path": "/etc/hosts"})}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "127.0.0.1 localhost"},
                {"role": "user", "content": "And now?"}]
        conv = openai_to_messages({"messages": msgs})
        msgs = conv[0] if isinstance(conv, tuple) else conv
        p = tpl.render(msgs, tools=tools, add_generation_prompt=True)
        check(p.startswith("<|start|>system<|message|>You are Beam"), "the prompt opens with Beam's system block", p[:80])
        check(p.endswith("<|start|>assistant"), "the prompt ends with the generation header", p[-60:])
        check("Reasoning effort: 512" in p, "no effort named: the template's default (512)")
        check("<|start|>developer<|message|># Instructions\n\nBe brief." in p, "the system message as the developer block")
        check("read_file" in p and "<|start|>assistant to=read_file" in p, "the tool and the past call are rendered")
        check("I should read it." in p, "a past turn's reasoning is kept (Beam's template asks for it)")
        check("to=assistant" in p and "127.0.0.1 localhost" in p, "the tool result is rendered")
        check("Reasoning effort: 128" in tpl.render(msgs, tools=tools, reasoning_effort="high"), "effort high -> 128")
        check("Reasoning effort: 8" in tpl.render(msgs, tools=tools, enable_thinking=False), "thinking off -> minimal (8)")
        check("Reasoning effort: 256" in tpl.render(msgs, tools=tools, reasoning_effort="xhigh"), "effort xhigh -> 256")

        def run(stream, thinking=True, chunks=True):
            parser = OutputParser(thinking=thinking, tools=tools, stream_tools=True, beam=True)
            ev = []
            pos = 0
            rnd = random.Random(7)
            while pos < len(stream):
                n = rnd.randint(1, 9) if chunks else len(stream)
                ev += parser.feed(stream[pos:pos + n])
                pos += n
            ev += parser.finish()
            reasoning = "".join(e.text for e in ev if e.kind == "reasoning")
            content = "".join(e.text for e in ev if e.kind == "content")
            calls = [e.call for e in ev if e.kind == "tool_call"]
            return reasoning, content, calls, ev

        # an answer: analysis then final (the server's prompt ended with <|start|>assistant)
        s1 = "<|channel|>analysis<|message|>The user greets me.<|end|><|start|>assistant<|channel|>final<|message|>Hello! How can I help?<|return|>"
        for chunks in (False, True):
            r, c, calls, _ = run(s1, chunks=chunks)
            check(r == "The user greets me." and c == "Hello! How can I help?" and not calls,
                  "answer: reasoning and content%s" % (" (streamed in pieces)" if chunks else ""), (r, c, calls))
        # a tool call after reasoning, two parameters (a string with newlines, an integer), and a nested object
        s2 = ("<|channel|>analysis<|message|>Need the file.<|end|><|start|>assistant to=read_file<|channel|>analysis "
              "<|constrain|>xml<|message|><rfl:parameter name=\"path\">\n/tmp/a b\nc.txt\n</rfl:parameter>"
              "<rfl:parameter name=\"max_lines\">\n20\n</rfl:parameter><rfl:parameter name=\"opts\">\n{\"x\": [1, 2]}\n</rfl:parameter><|call|>")
        for chunks in (False, True):
            r, c, calls, ev = run(s2, chunks=chunks)
            ok = (r == "Need the file." and c == "" and len(calls) == 1 and calls[0].name == "read_file" and
                  calls[0].arguments == {"path": "/tmp/a b\nc.txt", "max_lines": 20, "opts": {"x": [1, 2]}})
            check(ok, "tool call: name and typed arguments%s" % (" (streamed in pieces)" if chunks else ""),
                  (r, c, [(x.name, x.arguments) for x in calls]))
            starts = [e for e in ev if e.kind == "tool_start"]
            check(len(starts) == 1 and starts[0].call.name == "read_file", "tool call: announced while streaming")
        # parallel calls (<|end|> between them) and the turn ending at <|call|> without its token (the server strips the stop)
        s3 = ("<|channel|>analysis<|message|>Two reads.<|end|><|start|>assistant to=read_file<|channel|>analysis <|constrain|>xml"
              "<|message|><rfl:parameter name=\"path\">\n/a\n</rfl:parameter><|end|><|start|>assistant to=read_file<|channel|>"
              "analysis <|constrain|>xml<|message|><rfl:parameter name=\"path\">\n/b\n</rfl:parameter>")
        r, c, calls, _ = run(s3)
        check([x.arguments.get("path") for x in calls] == ["/a", "/b"], "parallel calls, the last closed at the end",
              [(x.name, x.arguments) for x in calls])
        # a parameter the model left open (the next one starts): it ends there
        s4 = ("<|channel|>analysis<|message|>x<|end|><|start|>assistant to=read_file<|channel|>analysis <|constrain|>xml"
              "<|message|><rfl:parameter name=\"path\">\n/c\n<rfl:parameter name=\"max_lines\">\n3\n</rfl:parameter><|call|>")
        r, c, calls, _ = run(s4)
        check(len(calls) == 1 and calls[0].arguments.get("max_lines") == 3 and calls[0].arguments.get("path", "").startswith("/c"),
              "an unclosed parameter ends at the next", [(x.name, x.arguments) for x in calls])
        # thinking off (the parser starts in content): Beam's analysis is still reasoning, not answer text
        r, c, calls, _ = run(s1, thinking=False)
        check(r == "The user greets me." and c == "Hello! How can I help?", "thinking off: the analysis stays reasoning", (r, c))
        # an answer, then more reasoning, then an answer part (several messages in a turn)
        s5 = ("<|channel|>analysis<|message|>a<|end|><|start|>assistant<|channel|>final<|message|>first<|end|>"
              "<|start|>assistant<|channel|>analysis<|message|>b<|end|><|start|>assistant<|channel|>final<|message|> second<|return|>")
        r, c, calls, _ = run(s5)
        check(r == "ab" and c.replace("\n", "") == "first second", "answer, reasoning again, more answer", (r, c))
        # literal markup in the answer that is not Beam's tokens stays text
        s6 = "<|channel|>analysis<|message|>q<|end|><|start|>assistant<|channel|>final<|message|>Use <b>bold</b> and x < y.<|return|>"
        r, c, calls, _ = run(s6)
        check(c == "Use <b>bold</b> and x < y.", "markup in the answer stays as written", c)
        self.assertEqual(failed, [])


if __name__ == "__main__":
    unittest.main()

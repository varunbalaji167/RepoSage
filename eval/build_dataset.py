"""Authoring script for dataset.jsonl. Labels written from the httpx source BEFORE
running RepoSage on any of these queries, so they can't be tuned to its output.

Fields: intent, gold (file.py::Class.name), must_mention (string or list-of-alternatives,
each verified against the gold source), target_files, expect (answer|abstain|clarify).
"""
import json, pathlib

C, R = "code_explanation", "risk_assessment"
Q = []
def add(cat, query, intent, gold=(), must=(), files=(), expect="answer"):
    Q.append(dict(id=f"q{len(Q)+1:02d}", category=cat, query=query, intent=intent,
                  gold=list(gold), must_mention=list(must), target_files=list(files), expect=expect))

# -- A. named method / function (explicit identifier in the question) ----------------------
add("named", "What does Client.request do?", C, ["_client.py::Client.request"], ["build_request", "send"])
add("named", "What does Client.send do?", C, ["_client.py::Client.send"], ["auth", "redirect"])
add("named", "Explain AsyncClient._send_handling_redirects.", C, ["_client.py::AsyncClient._send_handling_redirects"], ["redirect", "history"])
add("named", "What does BaseClient._build_redirect_request do?", C, ["_client.py::BaseClient._build_redirect_request"], ["method", "headers"])
add("named", "How does DigestAuth._parse_challenge work?", C, ["_auth.py::DigestAuth._parse_challenge"], ["nonce", "realm"])
add("named", "What does Response.raise_for_status do?", C, ["_models.py::Response.raise_for_status"], ["httpstatuserror"])
add("named", "What does URL.copy_with do?", C, ["_urls.py::URL.copy_with"], ["kwargs"])
add("named", "Explain Headers.multi_items.", C, ["_models.py::Headers.multi_items"], ["list"])
add("named", "What does Cookies.extract_cookies do?", C, ["_models.py::Cookies.extract_cookies"], ["jar"])
add("named", "What arguments does Timeout.__init__ accept?", C, ["_config.py::Timeout.__init__"], ["connect", "read", "write", "pool"])
add("named", "How does the get_environment_proxies function work?", C, ["_utils.py::get_environment_proxies"], ["proxy"])
add("named", "What does BasicAuth._build_auth_header do?", C, ["_auth.py::BasicAuth._build_auth_header"], ["basic"])
add("named", "What does Client._transport_for_url do?", C, ["_client.py::Client._transport_for_url"], ["mounts"])
add("named", "What does create_ssl_context do?", C, ["_config.py::create_ssl_context"], ["verify"])
add("named", "What does request_context do?", C, ["_exceptions.py::request_context"], ["request"])

# -- B. conceptual (no identifier named; retrieval must find it from meaning) --------------
add("conceptual", "How does httpx decide whether a redirect should change the HTTP method to GET?", C, ["_client.py::BaseClient._redirect_method"], [["see_other", "see other", "303"], "get"])
add("conceptual", "How are sensitive headers hidden when a Headers object is printed?", C, ["_models.py::_obfuscate_sensitive_headers", "_models.py::Headers.__repr__"], [["[secure]", "secure"]])
add("conceptual", "How does httpx strip the Authorization header when a redirect goes to a different origin?", C, ["_client.py::BaseClient._redirect_headers"], ["authorization"])
add("conceptual", "How does the client choose which proxy transport to use for a given URL?", C, ["_client.py::Client._transport_for_url", "_client.py::BaseClient._get_proxy_map"], ["mounts"])
add("conceptual", "How does httpx work out the charset from a Content-Type header?", C, ["_models.py::_parse_content_type_charset"], ["charset"])
add("conceptual", "How are Link headers turned into a dictionary?", C, ["_models.py::_parse_header_links", "_models.py::Response.links"], ["rel"])
add("conceptual", "How does digest authentication pick the quality of protection (qop)?", C, ["_auth.py::DigestAuth._resolve_qop"], ["auth-int"])
add("conceptual", "What error is raised if you access response.content on a streamed response before reading it?", C, ["_exceptions.py::ResponseNotRead", "_models.py::Response.content"], ["responsenotread"])
add("conceptual", "How does a relative URL get combined with the client's base_url?", C, ["_client.py::BaseClient._merge_url"], ["base_url"])
add("conceptual", "How does the client make sure base_url ends with a trailing slash?", C, ["_client.py::BaseClient._enforce_trailing_slash"], ["slash"])
add("conceptual", "How does httpx tell whether a hostname is an IPv6 address?", C, ["_utils.py::is_ipv6_hostname"], ["ipv6"])
add("conceptual", "How are query parameters merged between the client and an individual request?", C, ["_client.py::BaseClient._merge_queryparams", "_urls.py::QueryParams.merge"], ["merge"])
add("conceptual", "How are proxy URL patterns ranked against each other?", C, ["_utils.py::URLPattern.priority", "_utils.py::URLPattern.__lt__"], ["priority"])
add("conceptual", "How does httpx handle a 307 redirect, does the request body get reused?", C, ["_client.py::BaseClient._redirect_stream", "_client.py::BaseClient._build_redirect_request", "_client.py::BaseClient._redirect_method"], [["stream", "body"]])
add("conceptual", "Is the Headers class case-insensitive? Explain how.", C, ["_models.py::Headers.__getitem__", "_models.py::Headers.__init__"], ["lower"])

# -- C. risk assessment, file identified (by filename, by class name, multi-file) ---------
add("risk", "How risky is it to change _exceptions.py?", R, files=["_exceptions"])
add("risk", "Is it safe to modify _client.py?", R, files=["_client"])
add("risk", "What's the impact of touching _types.py?", R, files=["_types"])
add("risk", "How risky would it be to change the URL class?", R, files=["_urls"])
add("risk", "Is it safe to modify client.py and _models.py?", R, files=["_client", "_models"])
add("risk", "Which parts would be affected if I change Timeout in _config?", R, files=["_config"])
add("risk", "How dangerous is it to refactor _utils.py?", R, files=["_utils"])
add("risk", "Would changing DigestAuth break anything else?", R, files=["_auth"])

# -- D. risk intent but no file -> must ask, not guess -----------------------------------
add("clarify", "Is it risky to change anything?", R, files=[], expect="clarify")
add("clarify", "Which file is the safest to modify?", R, files=[], expect="clarify")

# -- E. ambiguous phrasing: surface cues point the wrong way -------------------------------
add("ambiguous", "What would break if I change URL.join?", R, files=["_urls"])
add("ambiguous", "Is Client.send safe to call concurrently from multiple threads?", C, ["_client.py::Client.send"], [])

# -- F. out of scope: the answer is not in the 9 indexed files -> should abstain ----------
add("out_of_scope", "How does httpx implement HTTP/2 connection pooling?", C, expect="abstain")
add("out_of_scope", "What does MockTransport do?", C, expect="abstain")
add("out_of_scope", "How does the httpx command line client format and print responses?", C, expect="abstain")
add("out_of_scope", "How does httpx encode a multipart/form-data request body?", C, expect="abstain")
add("out_of_scope", "Which compression algorithms can httpx decode, and how?", C, expect="abstain")
add("out_of_scope", "What is the capital of France?", C, expect="abstain")

out = pathlib.Path(__file__).parent / "dataset.jsonl"
out.write_text("\n".join(json.dumps(q) for q in Q) + "\n")
from collections import Counter
print(len(Q), "queries", dict(Counter(q["category"] for q in Q)))

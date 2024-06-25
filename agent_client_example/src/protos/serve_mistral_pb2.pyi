from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class QueryRequest(_message.Message):
    __slots__ = ("id", "modelIdentifier", "queryString")
    ID_FIELD_NUMBER: _ClassVar[int]
    MODELIDENTIFIER_FIELD_NUMBER: _ClassVar[int]
    QUERYSTRING_FIELD_NUMBER: _ClassVar[int]
    id: int
    modelIdentifier: str
    queryString: str
    def __init__(self, id: _Optional[int] = ..., modelIdentifier: _Optional[str] = ..., queryString: _Optional[str] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ("requestId", "response", "runTime")
    REQUESTID_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    requestId: int
    response: str
    runTime: int
    def __init__(self, requestId: _Optional[int] = ..., response: _Optional[str] = ..., runTime: _Optional[int] = ...) -> None: ...

from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DetectionRequest(_message.Message):
    __slots__ = ("frame_id", "classes", "image")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    classes: _containers.RepeatedScalarFieldContainer[int]
    image: bytes
    def __init__(self, frame_id: _Optional[int] = ..., classes: _Optional[_Iterable[int]] = ..., image: _Optional[bytes] = ...) -> None: ...

class DetectionResponse(_message.Message):
    __slots__ = ("frame_id", "results", "image")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    frame_id: int
    results: _containers.RepeatedCompositeFieldContainer[Result]
    image: bytes
    def __init__(self, frame_id: _Optional[int] = ..., results: _Optional[_Iterable[_Union[Result, _Mapping]]] = ..., image: _Optional[bytes] = ...) -> None: ...

class Result(_message.Message):
    __slots__ = ("label", "score", "xyxy", "xyxyn", "xywh", "xywhn")
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    XYXY_FIELD_NUMBER: _ClassVar[int]
    XYXYN_FIELD_NUMBER: _ClassVar[int]
    XYWH_FIELD_NUMBER: _ClassVar[int]
    XYWHN_FIELD_NUMBER: _ClassVar[int]
    label: str
    score: float
    xyxy: _containers.RepeatedScalarFieldContainer[int]
    xyxyn: _containers.RepeatedScalarFieldContainer[float]
    xywh: _containers.RepeatedScalarFieldContainer[int]
    xywhn: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, label: _Optional[str] = ..., score: _Optional[float] = ..., xyxy: _Optional[_Iterable[int]] = ..., xyxyn: _Optional[_Iterable[float]] = ..., xywh: _Optional[_Iterable[int]] = ..., xywhn: _Optional[_Iterable[float]] = ...) -> None: ...

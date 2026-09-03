//! The slice of the ONNX protobuf schema needed to read initializers.
//!
//! Only three messages and five fields, because the topology is compiled in
//! (see [`crate::yunet`]) and the model file is consulted for weights alone.
//! This replaces `tract_onnx::pb`, which came with the whole runtime.
//!
//! Protobuf ignores fields it does not know about, so declaring a subset parses
//! a complete ONNX file correctly — everything undeclared is skipped. The tags
//! below are from `onnx.proto` and are the load-bearing detail: a wrong tag
//! silently reads the wrong field rather than failing.

use prost::Message;

/// `onnx.ModelProto`, reduced to the graph.
#[derive(Clone, PartialEq, Message)]
pub struct ModelProto {
    #[prost(message, optional, tag = "7")]
    pub graph: Option<GraphProto>,
}

/// `onnx.GraphProto`, reduced to the initializers (the weights).
#[derive(Clone, PartialEq, Message)]
pub struct GraphProto {
    #[prost(message, repeated, tag = "5")]
    pub initializer: Vec<TensorProto>,
}

/// `onnx.TensorProto`, reduced to what a float initializer needs.
#[derive(Clone, PartialEq, Message)]
pub struct TensorProto {
    #[prost(int64, repeated, tag = "1")]
    pub dims: Vec<i64>,
    #[prost(int32, tag = "2")]
    pub data_type: i32,
    /// Set when the exporter stored floats individually rather than packed.
    #[prost(float, repeated, tag = "4")]
    pub float_data: Vec<f32>,
    #[prost(string, tag = "8")]
    pub name: String,
    /// The usual payload: little-endian f32 bytes.
    #[prost(bytes = "vec", tag = "9")]
    pub raw_data: Vec<u8>,
}

/// `onnx.TensorProto.DataType.FLOAT`.
pub const DATA_TYPE_FLOAT: i32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trips through prost to confirm the tags encode and decode as
    /// declared. A wrong tag would still round-trip here, so this only pins the
    /// declaration; `OnnxInitializerMap::load` reading the real model is what
    /// proves the tags match ONNX itself.
    #[test]
    fn the_reduced_schema_round_trips() {
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![TensorProto {
                    dims: vec![2, 3],
                    data_type: DATA_TYPE_FLOAT,
                    float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    name: "w".to_string(),
                    raw_data: Vec::new(),
                }],
            }),
        };
        let bytes = model.encode_to_vec();
        let back = ModelProto::decode(&*bytes).expect("decodes");
        assert_eq!(back, model);
    }

    /// Unknown fields must be skipped, which is what lets a three-message
    /// schema read a full ONNX file.
    #[test]
    fn unknown_fields_are_ignored() {
        // A message carrying tag 999 that our schema never declares.
        let mut bytes = Vec::new();
        prost::encoding::string::encode(999, &"ignored".to_string(), &mut bytes);
        prost::encoding::int32::encode(2, &DATA_TYPE_FLOAT, &mut bytes);
        let decoded = TensorProto::decode(&*bytes).expect("unknown fields are skipped");
        assert_eq!(decoded.data_type, DATA_TYPE_FLOAT);
    }
}

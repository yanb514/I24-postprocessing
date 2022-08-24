{
"$jsonSchema": {
    "bsonType": "object",
    "properties": {
        "collection": {
            "bsonType": "str",
            "description": "collection to be evaluated on"
            },
        "fragments": {
            "bsonType": "object",
            "description": "key: gt_id, val: corresponding stitched_ids"
            },
        "id_switches": {
            "bsonType": "object",
            "description": "key: stitched_id, val: corresponding gt_ids"
            }
        }
    }
}
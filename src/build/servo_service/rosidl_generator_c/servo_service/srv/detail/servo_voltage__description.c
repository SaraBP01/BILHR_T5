// generated from rosidl_generator_c/resource/idl__description.c.em
// with input from servo_service:srv/ServoVoltage.idl
// generated code does not contain a copyright notice

#include "servo_service/srv/detail/servo_voltage__functions.h"

ROSIDL_GENERATOR_C_PUBLIC_servo_service
const rosidl_type_hash_t *
servo_service__srv__ServoVoltage__get_type_hash(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x6f, 0x4d, 0xd5, 0x76, 0x9b, 0x29, 0x57, 0x20,
      0xe9, 0x9c, 0x5a, 0x75, 0xb1, 0xed, 0xea, 0x4b,
      0xa1, 0x7f, 0x42, 0xfe, 0xbb, 0xf3, 0x80, 0xd4,
      0x1f, 0x2d, 0xec, 0x03, 0x2a, 0x0d, 0x87, 0x16,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_servo_service
const rosidl_type_hash_t *
servo_service__srv__ServoVoltage_Request__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x0e, 0x12, 0x5b, 0x9a, 0x6b, 0x2c, 0x67, 0xdb,
      0x09, 0x10, 0x33, 0x95, 0xe8, 0xf0, 0xe1, 0x1e,
      0xcc, 0x72, 0xa9, 0x72, 0x08, 0x97, 0x7e, 0x70,
      0x69, 0x24, 0xdb, 0x47, 0x75, 0x9a, 0x1c, 0x7b,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_servo_service
const rosidl_type_hash_t *
servo_service__srv__ServoVoltage_Response__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0xfc, 0xb7, 0x03, 0x5d, 0x31, 0x06, 0x59, 0xc3,
      0x7a, 0x8e, 0x7d, 0x72, 0xa7, 0x0b, 0x00, 0xa0,
      0xcd, 0x18, 0x57, 0x46, 0xad, 0x8d, 0xd5, 0x9c,
      0x0c, 0xfc, 0x7b, 0xaf, 0x0c, 0xe9, 0x3d, 0xb1,
    }};
  return &hash;
}

ROSIDL_GENERATOR_C_PUBLIC_servo_service
const rosidl_type_hash_t *
servo_service__srv__ServoVoltage_Event__get_type_hash(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_type_hash_t hash = {1, {
      0x0f, 0xd1, 0x22, 0xbf, 0x0b, 0xbc, 0x27, 0xb1,
      0xbb, 0x26, 0x8c, 0xe0, 0x5f, 0x93, 0x88, 0xd4,
      0x8c, 0x5e, 0x32, 0xd4, 0x8c, 0xd4, 0x17, 0x36,
      0x22, 0x1e, 0x53, 0x49, 0x19, 0x27, 0x18, 0x4b,
    }};
  return &hash;
}

#include <assert.h>
#include <string.h>

// Include directives for referenced types
#include "service_msgs/msg/detail/service_event_info__functions.h"
#include "builtin_interfaces/msg/detail/time__functions.h"

// Hashes for external referenced types
#ifndef NDEBUG
static const rosidl_type_hash_t builtin_interfaces__msg__Time__EXPECTED_HASH = {1, {
    0xb1, 0x06, 0x23, 0x5e, 0x25, 0xa4, 0xc5, 0xed,
    0x35, 0x09, 0x8a, 0xa0, 0xa6, 0x1a, 0x3e, 0xe9,
    0xc9, 0xb1, 0x8d, 0x19, 0x7f, 0x39, 0x8b, 0x0e,
    0x42, 0x06, 0xce, 0xa9, 0xac, 0xf9, 0xc1, 0x97,
  }};
static const rosidl_type_hash_t service_msgs__msg__ServiceEventInfo__EXPECTED_HASH = {1, {
    0x41, 0xbc, 0xbb, 0xe0, 0x7a, 0x75, 0xc9, 0xb5,
    0x2b, 0xc9, 0x6b, 0xfd, 0x5c, 0x24, 0xd7, 0xf0,
    0xfc, 0x0a, 0x08, 0xc0, 0xcb, 0x79, 0x21, 0xb3,
    0x37, 0x3c, 0x57, 0x32, 0x34, 0x5a, 0x6f, 0x45,
  }};
#endif

static char servo_service__srv__ServoVoltage__TYPE_NAME[] = "servo_service/srv/ServoVoltage";
static char builtin_interfaces__msg__Time__TYPE_NAME[] = "builtin_interfaces/msg/Time";
static char service_msgs__msg__ServiceEventInfo__TYPE_NAME[] = "service_msgs/msg/ServiceEventInfo";
static char servo_service__srv__ServoVoltage_Event__TYPE_NAME[] = "servo_service/srv/ServoVoltage_Event";
static char servo_service__srv__ServoVoltage_Request__TYPE_NAME[] = "servo_service/srv/ServoVoltage_Request";
static char servo_service__srv__ServoVoltage_Response__TYPE_NAME[] = "servo_service/srv/ServoVoltage_Response";

// Define type names, field names, and default values
static char servo_service__srv__ServoVoltage__FIELD_NAME__request_message[] = "request_message";
static char servo_service__srv__ServoVoltage__FIELD_NAME__response_message[] = "response_message";
static char servo_service__srv__ServoVoltage__FIELD_NAME__event_message[] = "event_message";

static rosidl_runtime_c__type_description__Field servo_service__srv__ServoVoltage__FIELDS[] = {
  {
    {servo_service__srv__ServoVoltage__FIELD_NAME__request_message, 15, 15},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {servo_service__srv__ServoVoltage_Request__TYPE_NAME, 38, 38},
    },
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage__FIELD_NAME__response_message, 16, 16},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {servo_service__srv__ServoVoltage_Response__TYPE_NAME, 39, 39},
    },
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage__FIELD_NAME__event_message, 13, 13},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {servo_service__srv__ServoVoltage_Event__TYPE_NAME, 36, 36},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription servo_service__srv__ServoVoltage__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Event__TYPE_NAME, 36, 36},
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Request__TYPE_NAME, 38, 38},
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Response__TYPE_NAME, 39, 39},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
servo_service__srv__ServoVoltage__get_type_description(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {servo_service__srv__ServoVoltage__TYPE_NAME, 30, 30},
      {servo_service__srv__ServoVoltage__FIELDS, 3, 3},
    },
    {servo_service__srv__ServoVoltage__REFERENCED_TYPE_DESCRIPTIONS, 5, 5},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = servo_service__srv__ServoVoltage_Event__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = servo_service__srv__ServoVoltage_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[4].fields = servo_service__srv__ServoVoltage_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char servo_service__srv__ServoVoltage_Request__FIELD_NAME__ids[] = "ids";

static rosidl_runtime_c__type_description__Field servo_service__srv__ServoVoltage_Request__FIELDS[] = {
  {
    {servo_service__srv__ServoVoltage_Request__FIELD_NAME__ids, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_UINT8_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
servo_service__srv__ServoVoltage_Request__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {servo_service__srv__ServoVoltage_Request__TYPE_NAME, 38, 38},
      {servo_service__srv__ServoVoltage_Request__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char servo_service__srv__ServoVoltage_Response__FIELD_NAME__vin[] = "vin";

static rosidl_runtime_c__type_description__Field servo_service__srv__ServoVoltage_Response__FIELDS[] = {
  {
    {servo_service__srv__ServoVoltage_Response__FIELD_NAME__vin, 3, 3},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_INT32_UNBOUNDED_SEQUENCE,
      0,
      0,
      {NULL, 0, 0},
    },
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
servo_service__srv__ServoVoltage_Response__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {servo_service__srv__ServoVoltage_Response__TYPE_NAME, 39, 39},
      {servo_service__srv__ServoVoltage_Response__FIELDS, 1, 1},
    },
    {NULL, 0, 0},
  };
  if (!constructed) {
    constructed = true;
  }
  return &description;
}
// Define type names, field names, and default values
static char servo_service__srv__ServoVoltage_Event__FIELD_NAME__info[] = "info";
static char servo_service__srv__ServoVoltage_Event__FIELD_NAME__request[] = "request";
static char servo_service__srv__ServoVoltage_Event__FIELD_NAME__response[] = "response";

static rosidl_runtime_c__type_description__Field servo_service__srv__ServoVoltage_Event__FIELDS[] = {
  {
    {servo_service__srv__ServoVoltage_Event__FIELD_NAME__info, 4, 4},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE,
      0,
      0,
      {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    },
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Event__FIELD_NAME__request, 7, 7},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {servo_service__srv__ServoVoltage_Request__TYPE_NAME, 38, 38},
    },
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Event__FIELD_NAME__response, 8, 8},
    {
      rosidl_runtime_c__type_description__FieldType__FIELD_TYPE_NESTED_TYPE_BOUNDED_SEQUENCE,
      1,
      0,
      {servo_service__srv__ServoVoltage_Response__TYPE_NAME, 39, 39},
    },
    {NULL, 0, 0},
  },
};

static rosidl_runtime_c__type_description__IndividualTypeDescription servo_service__srv__ServoVoltage_Event__REFERENCED_TYPE_DESCRIPTIONS[] = {
  {
    {builtin_interfaces__msg__Time__TYPE_NAME, 27, 27},
    {NULL, 0, 0},
  },
  {
    {service_msgs__msg__ServiceEventInfo__TYPE_NAME, 33, 33},
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Request__TYPE_NAME, 38, 38},
    {NULL, 0, 0},
  },
  {
    {servo_service__srv__ServoVoltage_Response__TYPE_NAME, 39, 39},
    {NULL, 0, 0},
  },
};

const rosidl_runtime_c__type_description__TypeDescription *
servo_service__srv__ServoVoltage_Event__get_type_description(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static bool constructed = false;
  static const rosidl_runtime_c__type_description__TypeDescription description = {
    {
      {servo_service__srv__ServoVoltage_Event__TYPE_NAME, 36, 36},
      {servo_service__srv__ServoVoltage_Event__FIELDS, 3, 3},
    },
    {servo_service__srv__ServoVoltage_Event__REFERENCED_TYPE_DESCRIPTIONS, 4, 4},
  };
  if (!constructed) {
    assert(0 == memcmp(&builtin_interfaces__msg__Time__EXPECTED_HASH, builtin_interfaces__msg__Time__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[0].fields = builtin_interfaces__msg__Time__get_type_description(NULL)->type_description.fields;
    assert(0 == memcmp(&service_msgs__msg__ServiceEventInfo__EXPECTED_HASH, service_msgs__msg__ServiceEventInfo__get_type_hash(NULL), sizeof(rosidl_type_hash_t)));
    description.referenced_type_descriptions.data[1].fields = service_msgs__msg__ServiceEventInfo__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[2].fields = servo_service__srv__ServoVoltage_Request__get_type_description(NULL)->type_description.fields;
    description.referenced_type_descriptions.data[3].fields = servo_service__srv__ServoVoltage_Response__get_type_description(NULL)->type_description.fields;
    constructed = true;
  }
  return &description;
}

static char toplevel_type_raw_source[] =
  "uint8[] ids\n"
  "---\n"
  "int32[] vin";

static char srv_encoding[] = "srv";
static char implicit_encoding[] = "implicit";

// Define all individual source functions

const rosidl_runtime_c__type_description__TypeSource *
servo_service__srv__ServoVoltage__get_individual_type_description_source(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {servo_service__srv__ServoVoltage__TYPE_NAME, 30, 30},
    {srv_encoding, 3, 3},
    {toplevel_type_raw_source, 27, 27},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
servo_service__srv__ServoVoltage_Request__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {servo_service__srv__ServoVoltage_Request__TYPE_NAME, 38, 38},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
servo_service__srv__ServoVoltage_Response__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {servo_service__srv__ServoVoltage_Response__TYPE_NAME, 39, 39},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource *
servo_service__srv__ServoVoltage_Event__get_individual_type_description_source(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static const rosidl_runtime_c__type_description__TypeSource source = {
    {servo_service__srv__ServoVoltage_Event__TYPE_NAME, 36, 36},
    {implicit_encoding, 8, 8},
    {NULL, 0, 0},
  };
  return &source;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
servo_service__srv__ServoVoltage__get_type_description_sources(
  const rosidl_service_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[6];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 6, 6};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *servo_service__srv__ServoVoltage__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *servo_service__srv__ServoVoltage_Event__get_individual_type_description_source(NULL);
    sources[4] = *servo_service__srv__ServoVoltage_Request__get_individual_type_description_source(NULL);
    sources[5] = *servo_service__srv__ServoVoltage_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
servo_service__srv__ServoVoltage_Request__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *servo_service__srv__ServoVoltage_Request__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
servo_service__srv__ServoVoltage_Response__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[1];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 1, 1};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *servo_service__srv__ServoVoltage_Response__get_individual_type_description_source(NULL),
    constructed = true;
  }
  return &source_sequence;
}

const rosidl_runtime_c__type_description__TypeSource__Sequence *
servo_service__srv__ServoVoltage_Event__get_type_description_sources(
  const rosidl_message_type_support_t * type_support)
{
  (void)type_support;
  static rosidl_runtime_c__type_description__TypeSource sources[5];
  static const rosidl_runtime_c__type_description__TypeSource__Sequence source_sequence = {sources, 5, 5};
  static bool constructed = false;
  if (!constructed) {
    sources[0] = *servo_service__srv__ServoVoltage_Event__get_individual_type_description_source(NULL),
    sources[1] = *builtin_interfaces__msg__Time__get_individual_type_description_source(NULL);
    sources[2] = *service_msgs__msg__ServiceEventInfo__get_individual_type_description_source(NULL);
    sources[3] = *servo_service__srv__ServoVoltage_Request__get_individual_type_description_source(NULL);
    sources[4] = *servo_service__srv__ServoVoltage_Response__get_individual_type_description_source(NULL);
    constructed = true;
  }
  return &source_sequence;
}

set(FOLDER_NAME "Lab2")

set(SUBDIRECTORYES 
  01-vector-add-unified-memory
  01-vector-add-explicit-memory
  my_shared_vector_add
)

foreach(SUBDIRECTORY ${SUBDIRECTORYES})
  add_subdirectory(${SUBDIRECTORY})
  set_target_properties(${SUBDIRECTORY} PROPERTIES FOLDER ${FOLDER_NAME})
endforeach()

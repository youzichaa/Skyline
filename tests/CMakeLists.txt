macro (add_test_OT _name)
	add_executable(${_name}-OT "test_ring_${_name}.cpp")
    target_link_libraries(${_name}-OT SCI-OT) 
endmacro()

add_test_OT(skyline)
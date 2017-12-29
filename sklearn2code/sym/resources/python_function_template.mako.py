def ${function_name}(${', '.join(input_names)}, *args, **kwargs):
%for line in body_code.splitlines():
    ${line}
%endfor
    ${return_code}

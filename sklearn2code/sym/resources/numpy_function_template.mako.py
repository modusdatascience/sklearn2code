def ${function_name}(**kwargs):
%for name in input_names:
    ${name} = kwargs['${name}']
%endfor
%for line in body_code.splitlines():
    ${line}
%endfor
    ${return_code}
${function_name}.input_names = [${', '.join("'"+n+"'" for n in input_names)}]
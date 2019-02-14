def ${function_name}(kwargs):
%for input in inputs:
    ${str(input)} = kwargs['${str(input)}']
%endfor

%for rendered_assignment in rendered_assignments:
    ${rendered_assignment}
%endfor

    return ${', '.join(map(renderer.printer, outputs))}

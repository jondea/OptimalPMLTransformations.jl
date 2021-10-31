"""
    insertsorted!(v, el; by=identity)

Insert `el` into a sorted `v` at the correct sorted index.
"""
function insertsorted!(v, el; by=identity)
    i = searchsortedfirst(v, el; by)
    insert!(v, i, el)
end

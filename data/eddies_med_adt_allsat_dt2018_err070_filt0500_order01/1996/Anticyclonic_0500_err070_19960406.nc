CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��1&�        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�N�   max       P�Ѧ        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =�h        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F33333     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vyp��
>     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @O            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�             5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �T��   max       >R�        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�9n   max       B/��        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~N   max       B/�o        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C���        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�N�   max       PR��        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Y��|��   max       ?ޞ�v        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��t�   max       =�h        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F33333     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��(�\    max       @vyG�z�     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @M            �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @���            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B   max         B        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��䎊q�   max       ?ޚ�,<��        W�            |   <   0            �            <      =      .   /            1                  	         	         %      	                     +                  K                           
               GN���M�N�N���P\�O��|O�.�O^|NՍAOOSP�ѦO ofO���O O�YN�qP,;N��zO�K�O���N�K�O�2N��Pu��O��NNטND�VN~M�O�gBNd��NT�O�jWN���O_XNe��O�	�OA�N�/�N���OBYMO��O.POC�N�CO�غO��nM��>N�V9N_��N-O�E�OдN�t5N�:�OD��Ob\N��.O,^�O�4�NOg�Na=O <N��*N(�O+b;��t���o�T���#�
�#�
�o�o���
:�o;o;�o;�o;ě�;�`B<o<t�<t�<49X<u<�o<�C�<���<�9X<�j<ě�<���<�/<�`B<�h<�h<�<�<�<�<��=+=C�=C�=t�=t�=�P=#�
=#�
=,1=,1=8Q�=<j=<j=P�`=T��=Y�=Y�=Y�=Y�=aG�=e`B=m�h=q��=q��=�%=�o=�o=�-=�h/08<IPUWZZURI<70////[QUZ[[bghg[[[[[[[[[[-,+'&0<?CA=<50------����)Ngswsg[R+�����~zwvv��������������~����������������������������������������mnoqt|�����������tmmNMNQY[gt|���{tqg[ONN����)O������t[B��Z\chty��������thf^[Z��������������������xxz��������������zxxegnqz������������zneebagtu�����tngeeeeee��/;FKT`bSH;/"	��d^\chktuz|����}tqkhdUU\`XH</#
����
/<Uzvrtz��������������z)56<95) ��������������������HOT[ehnstph[[OHHHHHH���#05GT^\LH5)�|{����������������||
!

����������������������������������������fjt�������������pif�����
����������������������������������������������������������������������������
  
�����.))/5<CHLIH<5/......����)/.*#����
#03774.,'#
�������������������#/3:<B</#������	!" ��(&%$5BNU[_^`^[SB5/)(,'('/3<BHQUXVUPH</,,:64557:;<HT]be`URH<:��������������������
)-9?B@75) 
���)36;85.*)���7:<GHTMKH<7777777777���������()*)!*6?;6*�����
�������kjmz�������������zrk�������)5>@>85)�9/3;EHTWaahloomaTH;9�������������������������������������������������������(")6BFMFB64)((((((((��������������������}}�����������������������������������xxuyz��������zxxxxxx�������

�����μ4�8�@�B�I�M�P�M�@�4�*�'�"�$�'�2�4�4�4�4�5�B�N�[�\�[�N�D�B�A�5�5�5�5�5�5�5�5�5�5�Y�f�r�����������r�f�e�Y�U�Y�Y�Y�Y�Y�Y��)�B�O�V�bĄē�t�O�B�<�/�������������)�6�B�E�K�R�O�J�6���
�������ùϹ����'�2�-�'�����ܹù����������ü������������������������s�r�h�r�v���3�@�L�Y�^�e�k�l�e�`�Y�X�L�B�@�3�,�0�3�3���������������������������������������ż��������ڼ��ʼ����r�e�t�{�p�s�������� � ����
�������߽�������ݽ������(�9�:�4�(�������սͽɽ��T�a�b�m�q�z�~���z�z�m�a�T�R�K�H�I�S�T�TE�E�FFF$F'F$FFE�E�E�E�E�E�E�E�E�E�E�Ŀ��������������ĿĳıįĳĺĿĿĿĿĿĿ�"�;�a�m�����z�m�a�H�;�������������"���ʾ׾���پ׾ʾ����������������������m�`�T�;�.�'�(�(�0�G�Q�m�y�������������m�(�5�Z�g�x�������s�g�]�N�5��������(�Z�f�k�k�k�h�f�Z�M�L�K�M�N�V�Z�Z�Z�Z�Z�Z�s�������������������s�g�c�Z�U�Q�[�g�n�s�H�N�U�[�a�c�a�U�H�<�;�;�<�B�H�H�H�H�H�H�uƎƚ�������=�F�$�����Ƨ�u�\�W�X�\�u�����������������������������������������M�Z�\�\�Z�M�A�4�.�4�A�A�M�M�M�M�M�M�M�Màäìùû��ùîìëæàÛÖàààààààìöù����������ùìàÛÞàààààà�A�M�_�f�s�x�w�s�f�M�(�������(�4�AŭŲŹŽŹųŭŠŔŌŔśŠŬŭŭŭŭŭŭ�/�<�H�P�U�H�>�<�5�/�,�-�/�/�/�/�/�/�/�/�����Ŀѿݿ������ݿѿĿ�������������������	���������������������������������ʾ׾ݾ�پ׾ʾ��������������������s�����������y�s�f�f�b�f�r�s�s�s�s�s�s�a�n�zÄÊÌËÇ�z�n�a�U�H�<�7�0�2�<�H�a�Z�f�s�~����������������s�f�Z�M�C�D�O�Z�H�T�_�a�c�a�^�T�Q�H�H�;�7�3�8�;�=�C�H�H�"�/�;�B�?�;�/�,�"����"�"�"�"�"�"�"�"�/�;�H�T�a�o�q�m�i�a�T�H�;�/�(�"�"�'�,�/�����	�"�)�)�'�"���������������������������������������������������������������)�5�B�C�C�5�)�������������"�.�4�7�;�;�;�5�.�"��
����!�"�"�"�"�����(�5�:�8�5�(�������ؿѿĿݿ�����������źĺ������~�r�e�Y�R�X�e�r�~����ÇÓÔàâàÓÇÄÂÇÇÇÇÇÇÇÇÇÇ�F�S�_�l�l�v�l�a�_�S�S�S�F�:�/�2�:�B�F�F���!�,�'�!������������������`�l�q�u�r�l�`�\�[�`�`�`�`�`�`�`�`�`�`�`��'�*�+�(�������ܻٻһӻܻ߻���������������������ŹŭūŠŠŠşŠŭŹ�ƿ��������������������z������������������Ŀ������������������������ĿĽľĿĿĿĿ��#�0�1�8�8�5�0�#��
�����������
���ĚĦĳĹĿ����ĿĳĳĦĚčā�ĂċĔęĚŭŹ����������������Źŭũũŭŭŭŭŭŭ�����������������y�r�k�`�\�Z�`�k�m�y�~�������	�������	������������������������������������������������������������²¿��������������¿½²±®²²²²²²�ûлܻ���ܻۻڻлû»������������ûû��������������������޻��@�M�Y�f�g�f�Y�T�Q�M�G�C�@�>�@�@�@�@�@�@DoD{D�D�D�D�D�D�D�D�D�D{DyDoDjDdDbD`DbDo F ` ] n ( b * J ( d 8 = : ) O   | F ? h L V c @ N i k / \ * < J % b / Q f Q + <  F X : / a S G Z O \ @ 2 A U Z < N P h @ W b G  �  ;  �  Q  E    :    @    ]  �  a  ~  �  �     c  4  �  U  �  �  m  l  �  �    �  g  F  �  L  s  !  �    �  �  R  .  �  �  =  o  8  �  j  U  ~  v  �  �  �  �  �  �  y  n  �  ,  �  i  ��D���T���ě�=�l�=H�9=�w;�`B<#�
<���>R�<���=o<�1=�+<��
=�O�<�o=ix�=y�#<�j<�h<ě�=�hs=t�<�=o='�=ix�=�P=�P=m�h=�w=<j=t�=�C�=49X=0 �='�=e`B=aG�=�+=�o=<j=�1=��=D��=m�h=]/=Y�>J=�+=�o=��=�o=�C�=�%=���=�{=�C�=�7L=��T=���=�^5>?|�B&�TB�5B%�[B�mBl�B��B*��B�!B	*�B^RB]B!�fB ��B�B	�,A�9nB$1BB�MBq�B��BO�B��B��B{B!m�B?OBK2B��B��B�,B92B�B�0B��B%�B9�B%B[B��B��BF�B!#�B�B�B3B�B��B/��B�^B f)B�B�B\�A�Q!B}8B,UhB-�B�B��B��B!l�B��B��B&�fB��B&/wB�TBC�B��B*��B±B	2nB4}B<�B!��B ԲB�IB	��A�~NB�bB�~B��B��B@B}BB�4B��B4]B!��B@B�9B��B�gB��BBtB�B��B�B%2�B�B5�B?�B�B��BF�B!��B��B�B<^B�XB��B/�oB�	B u
B�!B�B��A� BM�B,EXB��B/AB��BA:B!��B@!B��@ϡ�A��@��dA��rA֤F>��@큷?Й�A�6�@���A0��A0�A���C���A�9cA�gAP3OAi�A��A?R�A��,A��lB�A��4A<9QA�+A�sLA9P1A��:AîrAy��A�'hANƾAD6�A��xAC��A��,A�Q.A��A�oA�CA��hA_2?A�$P@�A�U�@���@b�AB�@�zsA���Aq��A�fvA���A�ؼA�j�AXEA���A��8A��Y@���@�Æ@�U�C���@��
A���@��A׀�Aք�>���@�H,?��A��A@���A0��A0�A���C���A�A���AQ?Ak.A�lA>��A���A�EcB?�A��rA<�A�zBA�d�A8�A��2AÂ5Ax�IA�~ZAOCjACgA�~ AC>�A�~�A��A���A�VA��A�F�A^��A�k�@7�A�q�@��@\�ACL@��-A��Ar�A䁙A�wCA�d6A�t�A@�A�v;A���A��C@��j@�dO@��+C���            |   <   0            �            <      =      /   /            2                  	   	      
         &      
                     +             	      L                           
               H            7      #            =                  )      $   %            9               #                                                                                                                        #      !                              !                     5                                                                                                                           N���M�N�N���O��-O�7O���N��{N@��OćO�P�O5�O_�NО�O��+N�qO���N��O�O�|NTޛO�2NC�/PR��N��jNNטND�VNC�9OWxNd��NT�O���N���O_XNe��O.�zOA�N�/�N���O2(�O��N�O+H�N�CO*�6O�o�M��>N�D�N_��N-O�f�OдN�t5N�O�OD��Ob\N��.NQ��O���NOg�Na=N�QN��*N(�O+b;  <  �  �  �  �  ?  2  �  %  �  K  �  L  	�  �  �  �  �  /  �  '  f  =  �  �  �  [  Y  |  t    d  �  "  �  �    5  �  �  *  �  J  �  C  �  �    �  e    �    N  �  N  T  _    �  �  z  �  9��t���o�T��=,1<e`B$�  ���
$�  ;o=��;ě�<#�
<#�
<t�<o<�`B<#�
<���<���<�t�<�C�<��
<���<�/<ě�<���<�h=#�
<�h<�h=+<�<�<�=#�
=+=C�=C�=�P=�P=8Q�=,1=#�
=aG�=0 �=8Q�=@�=<j=P�`=aG�=Y�=Y�=]/=Y�=aG�=e`B=�7L=u=q��=�%=��=�o=�-=�h/08<IPUWZZURI<70////[QUZ[[bghg[[[[[[[[[[-,+'&0<?CA=<50------���)BN[__\UB8��������������������������������������������������������������stu�������tssssssssNNR[[\gtz���ytqg[PNN�����'6<CB9(���`\^ehtw��������tih``��������������������}��������������ghjnrz�����������zngebagtu�����tngeeeeee"/;JQTSNH;/"e`^fhitty{|tsmheeee��
/8BNPQMH</#
�yz����������������}y)0595)��������������������JOU[ghjqph[OJJJJJJJJ��(5OQ[\WJD5)����������������������
!

����������������������������������������{w|����������������{�����
����������������������������������������������������������������������������
  
�����.))/5<CHLIH<5/......������%&#��
#03774.,'#
�������������������#/3:<B</#������ !��'&&*5BN[^]_][QB51+)'-,./<HKSOHD<6/------;75668<<HQU\`c_UNH<;��������������������
)59<<85.)�����)26;74-)�7:<GHTMKH<7777777777����� �����()*)!*6?;6*������	�������kjmz�������������zrk�������)5>@>85)�9/3;EHTWaahloomaTH;9��������������������������������������������������������(")6BFMFB64)((((((((��������������������~~�����������������������������������xxuyz��������zxxxxxx�������

�����μ4�8�@�B�I�M�P�M�@�4�*�'�"�$�'�2�4�4�4�4�5�B�N�[�\�[�N�D�B�A�5�5�5�5�5�5�5�5�5�5�Y�f�r�����������r�f�e�Y�U�Y�Y�Y�Y�Y�Y��)�6�B�N�U�g�n�m�h�[�O�B�%��������)�6�?�B�D�F�B�>�6�)���������������$������ܹ��������������ùι�������������������������~�s�}�����L�T�Y�d�e�Y�L�I�@�?�@�F�L�L�L�L�L�L�L�L���������������������������������������Ѽ������ʼҼռӼʼ������������������������������������������������ݽ�������&�/�0�(�������۽ӽҽ��T�a�l�m�y�z�~�z�o�m�a�\�T�P�M�N�T�T�T�TE�E�E�FFF#F&F$FFE�E�E�E�E�E�E�E�E�E�Ŀ��������������ĿĳıįĳĺĿĿĿĿĿĿ�/�;�T�a�g�o�q�h�T�H�;�/�"������"�/���ʾ׾���׾׾ʾ����������������������m�y�����������y�`�T�G�;�6�5�6�>�G�T�]�m�N�^�j�n�k�a�T�N�5�(��������(�5�N�Z�f�h�h�i�f�b�Z�S�N�R�Y�Z�Z�Z�Z�Z�Z�Z�Z�s�������������������s�g�c�Z�U�Q�[�g�n�s�H�L�U�Z�a�b�a�U�H�?�>�D�H�H�H�H�H�H�H�HƎƚƸ�����/��������Ƨ�u�e�^�\�f�uƎ���������������������������������������˾M�Z�\�\�Z�M�A�4�.�4�A�A�M�M�M�M�M�M�M�Màäìùû��ùîìëæàÛÖààààààìîù����������ùìàÝàæìììììì�4�A�M�R�Z�`�Z�M�F�A�4�(����� �(�/�4ŭŲŹŽŹųŭŠŔŌŔśŠŬŭŭŭŭŭŭ�/�<�H�P�U�H�>�<�5�/�,�-�/�/�/�/�/�/�/�/���ѿݿ��������ݿĿ���������������������	���������������������������������ʾ׾ݾ�پ׾ʾ��������������������s�����������y�s�f�f�b�f�r�s�s�s�s�s�s�H�U�a�n�z�}ÅÇÇÇÃ�z�n�a�U�H�F�@�F�H�Z�f�s�~����������������s�f�Z�M�C�D�O�Z�H�T�_�a�c�a�^�T�Q�H�H�;�7�3�8�;�=�C�H�H�"�/�;�B�?�;�/�,�"����"�"�"�"�"�"�"�"�/�;�H�T�l�m�o�m�h�a�T�R�H�;�/�-�#�(�-�/���	��"�(�(�%�������������������������������������������������������������������)�5�5�@�5�3�)�������������"�.�4�7�;�;�;�5�.�"��
����!�"�"�"�"����!�(�2�.�"����������������~�����������úú������~�r�e�Y�Y�e�r�w�~ÇÓÔàâàÓÇÄÂÇÇÇÇÇÇÇÇÇÇ�F�S�_�j�l�t�l�_�S�L�F�>�:�4�:�D�F�F�F�F���!�,�'�!������������������`�l�q�u�r�l�`�\�[�`�`�`�`�`�`�`�`�`�`�`��'�)�*�'��������ۻӻԻܻ�����������������������ŹŭūŠŠŠşŠŭŹ�ƿ��������������������z������������������Ŀ��������������������ĿľĿĿĿĿĿĿĿ��#�0�1�8�8�5�0�#��
�����������
���ĚĦĳĹĿ����ĿĳĳĦĚčā�ĂċĔęĚŭŹ����������������Źŭũũŭŭŭŭŭŭ�y���������������y�x�m�w�y�y�y�y�y�y�y�y�����	������	��������������������������������������������������������������²¿��������������¿½²±®²²²²²²�ûлܻ���ܻڻջлûû������������ûû��������������������޻��@�M�Y�f�g�f�Y�T�Q�M�G�C�@�>�@�@�@�@�@�@DoD{D�D�D�D�D�D�D�D�D�D{DyDoDjDdDbD`DbDo F ` ] n   Y ( 7 & > 6 ? 8 ) O  t F 6 f L \ _ % N i r 7 \ * 7 J % b & Q f Q , ;  < X & ) a H G Z G \ @ 6 A U Z E G P h C W b G  �  ;  �  /    �  �  Z  %  i  #  �  �  g  �  �  �  p  ]  �  U  �    �  l  �  �  �  �  g    �  L  s  v  �    �  }  3  �  �  �  h  H  8  �  j  U  =  v  �  �  �  �  �  r  H  n  �    �  i  �  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  B  <  9  6  3  .  )  #    
  �  �  �  �  �  �  k  K  2       �  �  �  �  �  �  �  �  �  �  y  k  ]  L  1     �   �   �   �  �  {  p  c  U  F  ;  2  *  8  I  W  M  B  7  *        "  
?  d  @  �  m  �  9  p  �  j  0  �  '  �    !  
  �    �  �  c  �  ,  |  �  �  �  �  �  �  �  K  �  w  �  :  e  	  �  �    6  ?  =    �  �    h  �  z  U  #  �  j  �  k  2  *       (  .  2  2  /  )          �  �  �  �  S    �  �  E  r  �  �  �  �  �  �  �  �  �  �  �  �  �  h  C    �  �  �    #  $    	  �  �  �  �  �  t  S  4    �  �  �  �  �  �  �  (  �  �      �  _  �  �  �  [  �    '     �  f  *    E  I  K  I  C  8  %    �  �  �  �  �  �  �  q  V  8  	  �  @  s  �  �  �  �  �  �  �  �  �  �  �  g  <  	  �  �  �  Z    "  1  <  E  L  K  E  ;  -      �  �  �  {  Q  '  �  �  	�  	�  	g  	B  	  �  �  �  �  �  Q    �  l    �  +  �  X    �  �  �  �  |  b  B  !  �  �  �  �  v  A  	  �  �  O    �    x  �  /  j  �  �  �  a    �  y  )  �  W  �  O  �  �  x  �  �  �  �  �  �  �  �  �  u  c  R  C  3      �  �  �  �    I  z  �  �  �  �  �  �  �  �  s  )  �  Y  �  G  �  �  ,  �  �    )  .  -  (        �  �  �  c  
  �  #  �    �  �  �  �  �  �  �  �  �  �  �  �  h  ;    �  �    I     �  '    �  �  �  �  t  I    �  �  |  T  )  �  �  �  d  9    a  b  d  e  f  d  a  _  \  W  S  N  I  B  <  6  ;  B  J  Q    *  :  *    �  �          �  �  �  �  A  �  K  �   �  v  v  x  z  ~  �  �  �  �  r  [  =    �  �  �  �  V  3  #  �  �  �  �  �  �  �  �  �  �  �  �  {  r  h  _  `  �  �  �  �  �  p  P  !  �  �  �  f  j  v  {  V  1    �  �  �  h  >  S  W  Z  V  C  !  �  �  �  l  4  �  �  v  0  �  �  N  �  p        *  8  D  N  U  Y  Q  C  *  �  �  i    �  g    �  |  {  z  z  |  ~  �  �  �  �  �  o  T  !  �  �  �  I    �  t  r  o  i  `  W  K  @  -    �  �  �  o  6  �  �  v  1   �  �        �  �  �  �  �  �  �  ^  5    �  [  �  �    f  d  ]  W  P  J  C  <  0  !    �  �  �  |  F    �  �  P    �  �  �  r  `  L  6      �  �  �  �  W    �  �  ;  �  "  "      �  �  �  �  �  �  �  �  �  �  m  &  �  �  B   �   �    D  c  y  �  �  �  �  �  v  d  A    �  y     �  M    �  �  �  �  �  �  �  �  }  t  n  n  h  `  E  &  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  Z  <    5  +  "        �  �  �  �  �  �  �  p  Q  1    �  �  �  �  �  �  �  �  x  _  C  &  
  �  �  �  �  R    �  �  D  �  �  �  �  �  �  �  �  �  g  A     �  �  �  �  t  U  #  �  �  �  �      $  )  )  $    �  �  �  T    �  p    �  N  �  �  �  �  �  �  �  z  k  Z  D  +  	  �  �  U  �  �  (  �  �  J  @  6  ,       	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  J  k  �  �  �  �  �  �  m  ?     �  Q  �  '  o  �  �  �  8  A  ;  1  "  
  �  �  �  _  (  �  �  �  i  7  �  r  �    �  �  �  �  �  z  u  h  Y  I  9  *       �  �  p  A     �  �  �  �  �  �  �  w  ^  E  .      �  �  �        �  �          �  �  �  �  �  �  �  �  �  h  O  2    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  {  w  t  -  a  H  /    �  �  [  0    
�  
�  
A  	�  	i  �      �        �  �  �  �  �  �  p  J  "  �  �  �  f  5    �  .  �  �  �  b  @    �  �  �  i  7    �  �  ]  '  �  �  �  a  .  �    �  �  �  �  �  s  S  3    �  �  �  �  ]  ,  �  �  �  N  E  =  4  +  "    
  �  �  �  �  �  o  G  *    �  �  �  �  �  �  �  z  f  Q  ;  %    �  �  �  �  �  Y  +  �  A  *  N  G  A  9  +        �  �  �  �  �  �  �  �  �  �  �    �  �  �  �    (  6  B  L  S  T  R  B  $  �  �  �  {  S    Y  Z  H  2       �  �  �  t  C  	  �  w  "  �  j    �  �          �  �  �  �  {  U  *  �  �  �  u  J  (    �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  [  A    �  �  �  q  �  �  �  �  �  �  �  �  �  v  b  `  Y  G  /    �  �  �  �  z  r  n  i  a  Q  ,  �  �  �  C  �  �  k     �  �  �  �  �  �  �  �  u  a  I  "  �  �  �  �  �  r  b  R  A  /       �  9  �  �  �    �  �  a    �  F  �    ;  [  b  
Z  	G    
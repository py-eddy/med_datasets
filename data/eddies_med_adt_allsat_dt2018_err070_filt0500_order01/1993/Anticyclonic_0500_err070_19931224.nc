CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���"��`      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mԣ�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       >J      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @E������     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?޸Q�    max       @vt��
=p     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @P�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @�-�          �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��`B   max       >333      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B,��      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�t   max       B,F�      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?I��   max       C�w      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?P�   max       C�u�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          9      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          1      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mԣ�   max       Ps�]      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?ܪd��7�      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       >o      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?k��Q�   max       @E���R     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @vt��
=p     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @          max       @P�           p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�x        max       @��          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?ܠ�-�     �  Pl         	                           
   ,               y   ,            5            &            $               c            
             
               l                     N3t�Nڧ�N|�ZN(�~M�eN��VP�xN���OG04N�UN�k�O.A.O��N��NA�{N_��O�'�P��P��O;3NJ�N��O�e�O5�N,S�N.CO�s�O�U�OWU�N3��Ob�AN�5dO �LO+@�N���P X�O�T/N��>O*��N�eMԣ�O��VOmz]N�̣NBx6O3]kN'`�O1�O�4SN, N���N�zJN'^N���N��@N��y�+�e`B�T���D���o�ě��o��o;D��;��
;ě�;�`B;�`B<#�
<#�
<D��<D��<u<�1<�1<�9X<�9X<�9X<���<�=o=o=o=o=+=C�=\)=��=��=#�
=,1=,1=,1=8Q�=<j=<j=@�=@�=T��=T��=aG�=�o=�+=�C�=���=���=�{=�-=�9X=\>J!#./66/#!!!!!!!!!!��������������������SPO[gjtuttg[SSSSSSSSacht����thaaaaaaaaaa�������������������������	�������)5BPYFKE<1)������������������������������������������rpqssrt{����������tr)552-)tz�������������|xxtt  /<HUahqqaUI<7/#�����������������������������������������������������������������
#/543'#��gdelt�����������zztg����6BA6)���������������	���FHTaacfaTQHHFFFFFFFF����������������
6O[dimie[OB6���������������������������������������������������������� )5BFMOMJB5) %$&0@[gy�yysqg[NB5)%E><HLanz}�}zxnaUTHE����������������������������������������MJJO[hqtxxtlh[XOMMMM���������������������������


������������������������������
#!
��#/<U]gnongaUH;0 �������������������������������������������������������������������������
,6BOS[^ZOH?6)���������������� ���"$(	()*69BFJEB?62)((((((����%))+)���.,+/6<DG</..........a^]amtz}����zmkca_`a��������

�����)))&&&&)/5=BDDB>5)&&&&&&soptzz}������zssssssTVY[gnmge[TTTTTTTTTT���������������������~}������������������������������������D�D�EEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������������������������������L�Y�Y�_�Z�Y�L�E�A�I�L�L�L�L�L�L�L�L�L�L�'�-�3�4�4�3�'�#� �"�'�'�'�'�'�'�'�'�'�'��������������������ùôìèìõù�������	��;�X�`�_�Y�Y�T�H�;�/��������������	��������� ����������������������������A�N�Z�d�f�a�Z�N�H�A�5�-�(���$�(�5�6�A�n�zÇÓàìõùÿùìàÓÇ�}�z�n�l�j�n�׾����������׾ҾѾ׾׾׾׾׾׾׾��$�/�9�;�@�;�"��	���������������	�� �$��������!�%�"�$����������ü���������O�V�Z�[�c�[�T�O�B�:�6�2�5�6�B�I�O�O�O�O�h�tāąĄā�t�k�h�^�h�h�h�h�h�h�h�h�h�h���*�6�C�O�P�O�C�7�6�*���������;�G�T�`�m�y�����~�v�`�G�;�"����"�'�;�6�B�O�[�b�i�m�j�[�6�����������������6�H�T�X�V�M�;�2�!�	������������������/�H�"�.�;�C�G�T�W�Q�G�;�6�.�"���	��
��"�������ݿѿǿϿѿݿ޿�������꾘�����������������������������������������ʾ׾���������ʾ������������������������� �������ڽнĽ��Ľҽݽ��a�n�z�~�z�x�p�n�i�d�a�^�a�a�a�a�a�a�a�a�#�/�2�7�5�/�#���#�#�#�#�#�#�#�#�#�#�#�)�5�B�l�y�{�t�[�B�5�)���������)���������#�#������������ƼƷƵƶ��������"�(�?�D�;�5�(������������� ���#�/�<�E�<�/�.�#������������zÁÇàóù��ýùàÓÇ�z�r�q�z��z�y�z�������ƼǼļ����������������������������s�������������s�i�f�c�Z�W�W�Z�]�f�k�s�Z�f�s�����������s�f�Z�Y�M�H�J�M�P�Z�Z��*�.�2�.�*�"�!��	���������� �	���)�5�[�g�y��s�g�N�)��������������)�����������������������w�m�h�c�e�m�z����E�E�E�E�E�F	E�E�E�E�E�E�E�E�E�E�E�E�E�Eٺ����'�1�8�9�3�'��������������4�7�@�A�E�B�@�4�'��!�'�'�2�4�4�4�4�4�4������������������������~�����������źԺͺɺ����������r�a�Z�m�~�`�l�y�������������������y�l�e�_�W�Z�\�`�!�-�:�F�S�^�T�S�F�>�:�-�!������!�!���������������������������� �"��������������������b�o�{ǅ�{�x�o�b�Z�`�b�b�b�b�b�b�b�b�b�b��������������������ŹŭšşŠŭŲŹ����DoD{D�D�D�D�D�D�D�D�D�D�D�D{DlD_DaDaDjDo�0�<�A�I�I�I�C�<�8�0�*�+�0�0�0�0�0�0�0�0�b�n�{ŁŇŋŇŁ�{�n�g�b�X�W�b�b�b�b�b�b�M�Y�f�r�����|�r�f�Y�Q�M�I�M�M�M�M�M�M�A�N�Z�\�_�Z�N�D�A�5�A�A�A�A�A�A�A�A�A�A�ʼּ������������ּʼʼǼʼʼʼʼʼ�ÇÓàìù��ýùìàÛÓÇ�|ÇÇÇÇÇÇE�E�E�E�E�E�E�E�E�E�E�EuE�E�E�E�E�E�E�E� X - ; S x d 8 � 3 f H K * 5 9 � A 2 R 9 S f ! L h J < 2 , X 2 + 8 # M T P 2 + X s R Z \ V : Z Y % 2 4 / W B q 5  8  �  �  j  Y  �  {  N  �  H  �  �  q  �  R  �  �  �  u  L  w  D    �  n  ^    �  �  h  �  �  5  j  %  P  �  �  k  �  J  �  +    l  �  B  n  ;  N  �  �  H  �  �  ���`B�o��o���
���
;�`B<u;��
<�o<�1<T��<�o=P�`<�o<u<u=t�>C�=�o=o<ě�<���=��P=@�=t�=��=�O�=T��=ix�=�w=�\)=e`B=<j=]/=@�>\)=�\)=q��=u=e`B=H�9=���=�O�=y�#=e`B=��=�C�=��w>333=���=� �=���=�Q�=���=�/>hsB�B�mB	�B)�B!G\B��B+�BZB%B
B��B
��B�sB	�Bd�B�OBJyB
�PB�QB}�A��BG�BMgB"h�B��B��B�BRBQ1B��B!��B�xB$�B��B#+1BKBJBh�B0]B�3B![B��B,��BF|BۦB�B��A���B�B'�B/Bj�B��B��B LfBWlB�B��B	0�BA�B!*B�}B�BհB,)B
?B��B
��B��B�B?PB�zBDsB*B?�B�^A�tB>1BC�B"�B�B�B BE�B2B5�B"@B��B�B=�B#�BA�Be@BE�B	BQ�B!A1B?UB,F�B?�BϱB"JB�A�D1B�2B"B?�B?�B� B>tB ?�B>�C�F�?I��A�}�?ˌN?���A�R-A�Q�A���A��1A�+�AVL�A�$`AҲ�A��A�x#A��Ae7_A�\A��lAasA}7AJ��AP��A//A�\uA�]A�OUB��A�`�A��A��@��QAC �AA�
A[�A���A�z�C�w?{G�@͂8@���@�A�@wɭ@X@�݊B��A��FC��DA�&�A�@�c�A�X�AķA˂C�)C�I?P�A���?��?�A���A�sA�pxA��\A�t�AVX:A���Aҁ�A�ZMA��5A�+:AhA�TA�_A`�%A{'AI��AP��A/��AǁMA���A��yB�A��A��]Aʂ�@�AC�AA�A\�`A���A�;�C�u�?e{N@��F@�?�@
�IA�@t��@Z�@��6B�TA�eC��\A��A�Z�@��EA�=zA�	A˅�C�r         
                           
   -               z   ,            5            &            %      	         c                  !      
               l                                          '                              !   +   9            #         
   #                           +                  !                                                               !                              !      1                     
                                                !                                          N3t�N��,NI�7N(�~M�eN:��O��NOxBO1��N���N�k�O.A.N�v�N��NA�{N_��O�'�O2�WPs�]N�NNJ�N��OJzO�N,S�N.CO]]O��EO=��N3��O�MN�'�N��4O+@�N���O�c�O��!N��>O*��N�eMԣ�O�j�N�~�N�̣NBx6O3]kN'`�O1�O���N, N���N�zJN'^N���N��@Nn`l  �  1  4  �  �  Q  C  �  u  h  [  �  4  �  t    >  �    �  t  (    �  �  H  �  �  �    �  �  �  	    �  �  �  o  E  �  �  �  ^  [  �  �  w    �  �  �  �  u  �  q�+�D���D���D���o��o;o%   ;�o;�`B;ě�;�`B<���<#�
<#�
<D��<D��=���<���<�j<�9X<�9X=49X<�`B<�=o=<j=C�=C�=+='�=�w=�w=��=#�
=��
=0 �=,1=8Q�=<j=<j=L��=m�h=T��=T��=aG�=�o=�+=�\)=���=���=�{=�-=�9X=\>o!#./66/#!!!!!!!!!!��������������������WRS[ghtqg[WWWWWWWWWWacht����thaaaaaaaaaa��������������������������������5>525BDB;5)������������������������������������������tqrttttst���������tt)552-)tz�������������|xxtt***./<HJSUXUQH=<8/**�����������������������������������������������������������������
#/543'#��zwx������������������6=@6)�����������������FHTaacfaTQHHFFFFFFFF����������������'%%*6BOY[_a`\[QOB61'����������������������������������������������������������)5<BDFGFB:5)&((.3B[gv|wvpl[NB5)&HBEHOUanz|~|zwnaZUH����������������������������������������QOS[hntuuth[QQQQQQQQ���������������������������


�����������������������������

����!#/<HU\fmoneaUH<1!�������������������������������������������������������������������������

26BOX[WOKF=6)
�������������������� ���"$(	()*69BFJEB?62)((((((����%))+)���.,+/6<DG</..........a^]amtz}����zmkca_`a��������

�����)))&&&&)/5=BDDB>5)&&&&&&soptzz}������zssssssTVY[gnmge[TTTTTTTTTT���������������������~}������������������������������������D�D�EEEED�D�D�D�D�D�D�D�D�D�D�D�D�D�����������������������������������������������������������������L�Y�Y�_�Z�Y�L�E�A�I�L�L�L�L�L�L�L�L�L�L�'�-�3�4�4�3�'�#� �"�'�'�'�'�'�'�'�'�'�'ù��������������ùùììì÷ùùùùùù�	��;�R�T�P�H�C�;�4�0�"��	����������	���������������������������������������A�N�Z�b�e�`�Z�V�N�D�A�5�2�(�!� �&�5�8�A�n�zÇÒÓàáìðöìàÚÓÇÁ�z�n�m�n�׾����������׾ҾѾ׾׾׾׾׾׾׾��$�/�9�;�@�;�"��	���������������	�� �$������������ ���������������������O�V�Z�[�c�[�T�O�B�:�6�2�5�6�B�I�O�O�O�O�h�tāąĄā�t�k�h�^�h�h�h�h�h�h�h�h�h�h���*�6�C�O�P�O�C�7�6�*���������;�G�T�`�m�y�����~�v�`�G�;�"����"�'�;��)�6�B�O�P�V�U�O�L�B�6�)��������/�H�P�T�R�I�;�*��	������������������/�.�;�<�G�R�L�G�;�.�'�"� �����"�&�.�.�������ݿѿǿϿѿݿ޿�������꾘�����������������������������������������ʾ׾�������׾ʾ��������������������������
������нŽн׽ݽ��a�n�z�~�z�x�p�n�i�d�a�^�a�a�a�a�a�a�a�a�#�/�2�7�5�/�#���#�#�#�#�#�#�#�#�#�#�#�)�5�B�N�X�i�m�g�`�[�N�B�5�%������)����������� � ������������Ƽƺƾ���������(�/�<�B�9�5�(���	������������#�/�<�E�<�/�.�#�����������àìùÿùùìàÓÇÃ�z�v�v�zÁÇÓØà�������¼¼������������������������������s�|�������������s�f�[�Z�Z�Z�^�f�o�s�s�Z�f�s�����������s�f�Z�Y�M�H�J�M�P�Z�Z��*�.�2�.�*�"�!��	���������� �	���)�5�N�[�g�k�h�_�[�N�B�5�)���� ���)�������������������������y�m�j�d�f�n�z��E�E�E�E�E�F	E�E�E�E�E�E�E�E�E�E�E�E�E�Eٺ����'�1�8�9�3�'��������������4�7�@�A�E�B�@�4�'��!�'�'�2�4�4�4�4�4�4������������������������~�������������кǺ������������r�d�\�o�~�y�}���������������y�s�n�l�f�l�v�y�y�y�y�!�-�:�F�S�^�T�S�F�>�:�-�!������!�!���������������������������� �"��������������������b�o�{ǅ�{�x�o�b�Z�`�b�b�b�b�b�b�b�b�b�b��������������������ŹŭšşŠŭŲŹ����DoD{D�D�D�D�D�D�D�D�D�D�D�D{DmD`DbDcDkDo�0�<�A�I�I�I�C�<�8�0�*�+�0�0�0�0�0�0�0�0�b�n�{ŁŇŋŇŁ�{�n�g�b�X�W�b�b�b�b�b�b�M�Y�f�r�����|�r�f�Y�Q�M�I�M�M�M�M�M�M�A�N�Z�\�_�Z�N�D�A�5�A�A�A�A�A�A�A�A�A�A�ʼּ������������ּʼʼǼʼʼʼʼʼ�ÇÓàìù��ýùìàÛÓÇ�|ÇÇÇÇÇÇE�E�E�E�E�E�E�E�E�E�E�ExE�E�E�E�E�E�E�E� X 3 > S x X ( � 0 d H K  5 9 � A  N 3 S f  = h J 1 2 + X  % , # M 4 J 2 + X s V K \ V : Z Y $ 2 4 / W B q 1  8  �  U  j  Y  k  �    y  �  �  �  �  �  R  �  �  r    �  w  D  �  k  n  ^  �  �  �  h  P  �  �  j  %  9  t  �  k  �  J  R  �    l  �  B  n  /  N  �  �  H  �  �  x  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  c  O  7     	  �    %  /  0  .  %      �  �  �  �  �  �  �  �  �  y  3      2  A  O  S  T  M  B  5  &      �  �  �  �  �  �  m  �  �  �  �  �  �  �  �  �  �  �  �  �    i  U  A  C  N  Y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  n  e  \  3  *  5  P  <  (      �  �  �  �  �  �  �  �  }  p  Y  7  -  $  -  8  @  A  6  "    �  �  �  z  ^  A    �  �  �  '  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  (  Z  �  n  r  t  s  o  h  `  U  E  3    �  �  �  v  B  
  �  n   �  ?  Z  e  d  ^  P  -  �  �  d    �  d    �  '  �  �  a  #  [  T  L  D  ;  1  '        �  �  �  �  �  �  z  W  0    �  �  |  y  w  v  v  t  p  f  W  G  7  -  .  +      �  �  �  �  _  �  �       .  3  +      �  �  �    �    \  �  �    |  y  w  t  q  n  k  h  d  a  ]  Y  U  P  I  2      t  x  }  �  �    x  r  f  Q  =  (      �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  o  \  I  5    	   �   �   �   �  >    �      �    %    
  �  �  �  �  �  V  $  �  �    �  	�  
�    #  �  %  �  �  �  �  �  �  J  �  �  
�  	Z  r  �  �  �    �  �  �  �  `    �  �  �  �  �  �  b    �  �  6  h  u  ~  �  ~  y  q  h  Y  H  3      �  �  �  �  �  �  �  t  k  a  W  M  C  9  /  &              �   �   �   �   �   �  (  $             
     �  �  �  �  �  �  �  �  �  �  �  �  �  >  �  �  �  �        �  �  �  e    �    f  U   �  �  �  �  �  �  �  o  V  :    �  �  �  |  O    �  �  v  K  �  �  �  �  �  �  �  �  y  _  ?    �  �  �  [  &  �  �  �  H  E  B  ?  2  $      �  �  �  �  �  �  �  �  �  �  o  E  k  �  �  �  �  �  �  �  �  �  �  �  K    �  w  +  �  �    ~  �  �  |  k  X  F  3  !    �  �  �  �  p  A     �  u   �  �  �  �  �  �  �  �  u  S  *  �  �  �  `    �  �  E    �                              K  w  �  �  /  t  P  �  �  �  �  �  �  �  ^    �  �  R    �  p  �  T  �  �  �  �  �  �  �  �  �  �  x  a  E  !  �  �  �  �  �  �  {  O  w  �  �  �  �  �  |  r  b  Q  >  (    �  �  �  �  m     �  	     �  �  �  �  �  �  �  r  ^  @    �  �  a    �  �  C            �  �  �  �  �  �  �  �  �  y  h  V  F  6  &  	�  
a  
�  
�  1  a  �  �  �  o  5  
�  
�  
,  	�  �  �  �  ^  �  �  �  �  �  �  �  �  �  �  w  h  S  8    �  �  l     �  �  �  �  ~  i  S  :  #      �  �  �  �  �  r  [  D  )    �  o  j  d  [  N  ;  $    �  �  �  �  [  .  �  �  �  K    �  E  @  ;  2  (    	  �  �  �  �  _  1    �  �  m  8  �  �  �  �  �  �  �  ~  u  i  \  N  A  3  &      �  �  �  �  �  �  �  �  �  x  X  6    �  �  �  �  �  �  N  �  �  �  �    q  u  x  |  �  �  �  �  �  �  �  �  �  |  T  %  �  �  �  S  ^  T  J  <  .      �  �  �  �  k  5      *  7  ?  !    [  V  R  N  J  E  >  8  1  *         �  �  �  c    �  �  �    m  Y  >    �  �  �  e  +  �  �  f  #  �  �  �  _  |  �  �  �  �  �  ~  w  p  i  c  \  U  N  H  A  *    �  �  �  w  F      �  �  �  �  �  g  E  $    �  �  �  �  �  "  �  n  o  w  y  p  V  9    �  �  3  �    S  �  �  �  
�  	@  8  �  �  �  �  �  �  �  �  �  �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  d  L  4      �  �  �  w  D  �  �  �  �  �  �  �  �  q  N  "  �  �  �  ?  �  �  L   �   |  �  �  �  �  �  {  q  g  \  R  H  >  3  &    �  �  �  �  �  u  a  S  L  <  ,         �  �  �  �  h  ;    �  �  v  A  �  �  {  \  9    �  �  �  �  ^  1  �  �  d    �  p  	  �  h  o  n  j  b  R  8      �  �  �  u  8  �  �  Z    �  Q
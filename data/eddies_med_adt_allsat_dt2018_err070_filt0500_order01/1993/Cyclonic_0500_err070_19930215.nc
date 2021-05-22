CDF       
      obs    M   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?����+     4  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P`s�     4  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��O�   max       =�w     4      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @F��
=p�       !H   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vh(�\       -P   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @P�           �  9X   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�h�         4  9�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���#   max       <�j     4  ;(   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B1��     4  <\   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0��     4  =�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <^4�   max       C�&�     4  >�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >Bw�   max       C��     4  ?�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ^     4  A,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5     4  B`   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     4  C�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PG;j     4  D�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��e+��   max       ?фM:��     4  E�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���P   max       =�w     4  G0   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?\(�   max       @FxQ��       Hd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �У�
=p    max       @vh(�\       Tl   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @P�           �  `t   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��         4  a   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�     4  bD   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?X�t�j~�   max       ?фM:��       cx                           	      ]                  ;      .                                    "   	      	   	   0         	      >      #   !         \      	                              +               	      
                  )         (      N�{sN&�NH�3N��_N�/OuO���N#<TNzJ�O2��P`s�P��O��\N���M��O=O�0N�O�NA�ON1�O�NjZyN�۵OY�No�[N�h<N��N�EuND�9P Q2O[�O��mN��`N��O��VN��|NӘ�Nc�NL��P"�O�(|PG;jO�:�O"3OuإPQO:W-N�פNQ�|O5��N��Oq�MN�l,Oc�2O�DVO��SO%��O�y<N:��N�P�O)ֱN'
Np�(N@��N�Y�N�^N��O|}O��N��Of�NQ�O&c"O���NG�N�=�w<D��;�`B;�o��o�D����o��o���
���
�ě���`B��`B�#�
�#�
�49X�49X�D���D���D���D���D���e`B�e`B�u�u��o��C���C���C���t����㼛�㼛�㼛�㼛�㼣�
��1��j��j��j�ě�������/��/��`B���o�o�o�C��\)�\)�\)�\)�t��t���P����w�'0 Ž0 Ž0 Ž49X�49X�49X�8Q�@��P�`�aG��q���q���y�#��o��o��O��),13.)����EHUUabaUMHCAEEEEEEEE����������������}����������������~}}#&&#

#*/2674/(#
���
#<QTSN@</#
���mnz������znhmmmmmmmm),./)bgit����������tngb_b�������.22)������jsz���������������tj(,5BN[ckruo[NBAB66/(��
#+06<0#
��������������~����������
#/:@A@<8/#

NW[t������������t[QN+/2;><;/'"++++++++++��#,-<KQL48/*
��#/:7/#�������������������������������������
#,/7/#
O[`ht~�����toh^[OOOOFHJTampzwurmaTRIHEDF()/6BDJOOOB6+)((((((Z[_hhpt�����zth[ZZZZ���$'#�����15BNW[_a[NB5,-111111sz�������������zssss$3C\{����u\OC6*" $#$������������������������������������������ ������������������������������69;<866)�����zz����������zroozzzz )5BGNSTNDB52)�����������������������������������������������������������������#%���������#0Ubn�����{b<0
������������������|�gjrt������������~nhg��������������������|��������������}vqs|�������������������������������������&).5;BNSNMB65)&&&&&&������������������������������������������������������������ptw�������������~tpp���������������������$*!������}�����������������|}<<=@IUZbdggbZUI?<58<pt����������������upS[egntwtkge[YVSSSSSS���������������������������������������st|���������}tssssss������������������������������������������������������������o{�����������{tqoooo��������������������������	
����������#0<IUbjlibUP<,"�� 

����������� 
$&#"
�����#*/40/# ru����������unnpuqrUanz��������znaUPMNU������������������������

��������ֺ˺ɺ����������ɺֺ�����ݺֺֺֺ�ùñùù����������������ùùùùùùùù�[�Q�[�b�g�t�z�t�g�[�[�[�[�[�[�[�[��z�v�r�n�n�r�����������������������àÝÜàìõùùùìàààààààààà������������������������
�������������������#�)�6�B�O�h�o�k�\�]�O�?�1�)��H�H�D�B�D�H�L�S�U�Y�W�U�H�H�H�H�H�H�H�H�������������������������������������������������������������������������������˼ʼ������������ͼ����!�+�3�2�%�����������������ĶĴ�������
�/�<�0�(��
����M�A�<�:�:�?�M�Z�f�s�����������s�f�Z�M���������������������ʼ̼ϼϼмʼȼ������ݽؽн̽нݽ����ݽݽݽݽݽݽݽݽݽ��U�N�H�<�9�3�0�<�H�U�a�g�n�v�{�y�n�a�^�U���������������������ûлջݻݻڻӻû�����������������������������������������²¬¦¦²��������������������²�z�t�o�k�tàÙÓÇ�z�u�wÇÓàìùþ��������ùìà�Y�O�@�4�0�'��'�4�@�Y�f�n�r�z�z�r�h�f�Y�ݽڽ۽ݽ���������������ݽݽݽݽݽ��<�4�/�-�(�/�3�<�H�U�W�Y�U�O�H�C�<�<�<�<������ƳƮƲƳ�����������������������̻l�j�_�Z�]�_�l�w�x�z���}�x�s�l�l�l�l�l�l�����������������������������ýĽ�������ƳưƧƛƚƗƖƚƧƳƹ������������ƸƳƳƎƌƉƇƎƒƚƧƱƲƳƫƧƚƎƎƎƎƎƎ�;�8�/�.�/�6�;�E�H�T�T�T�N�I�H�?�;�;�;�;�ʾ����������ʾ׾������!���	��׾�ƧƜƚƎƍƎƕƚƝƧƳƾ��������ƿƳƧƧ���������������������$�*�2�4�4�0�$��ѿ̿Ŀÿ��Ŀѿݿ�������������ݿѿ��t�p�h�`�]�h�tāĈćā�y�t�t�t�t�t�t�t�t�Z�M�C�A�4�.�I�M�Z�f�������������s�f�Z��������������������������������������������������������������������������{�w�t�x�{ǈǊǔǘǘǔǈ�{�{�{�{�{�{�{�{�6�5�6�:�C�O�\�\�]�\�O�C�6�6�6�6�6�6�6�6�l�f�F�-���+�l�����л��������л����l�y�u�q�z�������������ĿѿӿпĿ��������y�������h�_�Q�O�R�g�s���������������������Z�N�A�8�5�0�1�C�N�^�g�s�����������s�g�Z�����s�p�g�i�s�y�������������������������������������������������������������������r�Y�M�Y�j�~�����ֺ���������ֺ���������ŹŰŭšŠŜŠŭŹ�������������������{�p�q�q�y�{ņŇňŊőŔŔřŔŇ�{�{�{�{����������������������������������������������������������������������������������������������������������������������t�n�g�m�s�����������������������������	��������������	���������	�	���׾ʾ������������ʾ׾������ �����Ç�z�n�i�j�o�tÇÓàãéìú����ùìàÇ�����������������#�0�<�I�N�O�O�I�0����������ݽӽֽݽ������(�*�'���������$�3�;�=�G�Y�r���������~�r�Y�L�4�'��;�4�;�?�H�O�T�_�T�R�T�[�T�H�;�;�;�;�;�;�s�k�g�_�f�g�s�������������������s�s�s�s�.�%�$�.�3�;�G�T�`�m�y�{�y�v�m�`�T�G�;�.���	���	�����"�#�"����������	���	��"�"�*�(�"���������������$�/�$�!�����������V�O�I�=�2�=�A�I�V�b�o�p�t�o�i�b�V�V�V�V�'����� �'�4�@�A�@�?�@�E�@�4�'�'�'�'����������������)�2�6�?�6�1�)�����������������Ľݽ���߽߽��ݽнĽ������������������л�������ܻû������m�l�m�p�u�z���������}�z�m�m�m�m�m�m�m�mEVEPECE7E*E!E'E*ECEPE\EuE�E�E�E�E�E�EuEVE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����������	��#�/�<�@�H�L�H�<�/�#��
���񹨹������������ùϹܹ�������۹Ϲù������������������ùȹù¹�����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� , X N ; L ; X � & ? A G ; V F  A & 4 a S G a U 3 ; W D > s G ) 8 A ; + W l / 6 � 5 * 2 / - G J ^ ^ * N : | A % > 3 ] g / g e Z x 3 2 1 Y D Z c U S # O Q      G  p  )    \  �  �  �  �    �    9  	  �  Y  4    �  �  d  �  �  �  �  �      �  �  %      �  �  �  ?  k  r  6  �  V  K  r  �  �  �  ,  �  }  �  �  L  �    L  \  �  r  �  �  �  �  �  �  �  !    �  `  !  7  �    Q  <�j<t�%   �49X��o�e`B��h�49X�T����1�����+��9X��9X�u�C���\)��t��ixռ�j��󶼼j��C����ͽ\)������j��`B��`B���
�P�`��`B����`B��`B��+����w����/��{�D���y�#�q���'e`B���#�e`B�'�w�P�`�#�
�}�'e`B�Y�����H�9���T�49X�]/�aG��D���P�`�L�ͽ]/�L�ͽu��o��7L�u���ͽ�+���T��񪽑hs��9XB�iB��B�mB��Bn_B�IB?�Bq�BUB
.�B-5|B@BUhB$��B)�XB�>B!A���B|bB�BP�B#�Bp�B�A�t'B�B+�B��B�BݵB1��Bi7B��BHBX�BX1B�BmnB$cB�cBK�B��B&�BeB�HB�~B�OBH�B�1Bx�B}�B�/B�_B
�iB=�BB�&B'sB �:B	OBc4B�B��B!&sB�B�B)�B�@B"�fB&��B�#B�"BۉB
t]B��B3�B5KB��B�mB�GB��BV�B�<B<.B�CBE B	ֽB-C0B%'B��B$ȳB)�YB�ABKA��B��B��B��B"��BAqB>�A��*B�FB?#BG�B?�B�FB0��B~�B��B?�B��B9=B��B��B4BB�B�]B��B&��B?TB�B��B��B?�B
8B@B?pB�BB��B
\�B�_B�CBȵB')@B D�B��B@4B?�B��B!C:BELB��B(�UB�B"EB&�,B��B>6B-�B
�JB~�B?�BU�@:;�A�gA��x@�l�A�s.A���A�J�AĞ�A��CA��BA	A���A@.{@��A+~�AŖ�@��zA�vA�YA�:;A�}@@ا�A,�}A���B�d@�x�A"}�B9 Bb�A�}$AT�B�B		�A}��A�]�AB��A���A�)�B��B ��@���Asj�A��;A�\�A�Z�A���@#�MA��ZA�KAqIuA���A�ےA�U�A[g�AT` A�|�A�#A0�?Հ�A���A�2�Af��A�w�A��[B	A�B�@�EnA�.$A(m-@�U�A�[C��kC�&�A��R>fu�<^4�C�k@:�A΁�A��@��ÀeA���Aׯ�A�voA�K�A�N�AA�N=AA�@��A+��A�l@� �A�#A��=A�|�Ǎ>@ڊ�A-��AÐ�B��@��9A"�B@mB@�A���AT
�B�uB	�A~��A�{�AB��A�z!A�q!B�OB%@��At��A�<�A���A���A�q@$*A���A�:fAp�A�O�A�j,A�s�A[�AS?iAɆA�i�A/�i?�EA�L{A�u7Ah��A�zA��)B	1�B!*@�!zA��A&�L@��A��PC��pC�(1A�Ǩ>Bw�C��C��                           	      ^                  <      .                                    "   
      
   
   0         	      ?      $   !         \      
                              ,               	                        *         (                                       5   /               %      '                                    %               %               5   !   -            )                              %      +                                 '                                                      %                                                            %                              '   !   -                                                                                 !                     N�{sN&�NH�3Ndb�N�/OuO2jN#<TNzJ�O�O�sOd+O"��N���M��N�b O��N�O�h�NA�ON1�N��GNjZyN�%yOY�No�[N��lN��N�EuND�9O�}bO[�O\�pN��`N��O:N���N��bN:BHNL��O�qsO�(|PG;jO��.OUO75�O�տO:W-N�פNQ�|N���N��Oq�MN��tOc�2O�DVOZQ�O��O�mN:��N�P�O)ֱN'
Np�(N@��N�Y�N�^N��O|}O���N��Of�NQ�N�w�O%g�NG�N��.  �  r  :  �  i  �    4  �  c    /  �  u  �  �  �  �  t  �  }  �  �  �  5  �  �  :  W  �  W    �  G  i  ,  �  �  �  M  �  T  �  �  {  �  
�  R  Y  �  �  �  4  �  �  �  �  8  ?  �  J  �  `  �  �  �    �  Z  �  b  
�  �  �  r  7  <=�w<D��;�`B�o��o�D���u��o���
�o�t���t��49X�D���#�
����ě��D����j�D���D���T���e`B��o�u�u��C���C���C���C���9X���㼴9X���㼛��#�
��9X��j�ě���j�t��ě�������`B��`B�+��o�o�o�o��w�\)�\)�t��\)�t��<j���H�9��w�'0 Ž0 Ž0 Ž49X�49X�49X�8Q�@��T���aG��q���q����o���P��o��\)�),13.)����EHUUabaUMHCAEEEEEEEE������������������������������������#&&#

#*/2674/(#
�#/<<HHJHE?<0/#mnz������znhmmmmmmmm),./)egmt����������{tgebe�����"#���������������������������45ABN[^fglng[NKBA<74��
#)/-#
�����������������~����������!#'/7:7/-#[[_gt�����������th[[+/2;><;/'"++++++++++��
#(1<AGB<.#

���#/:7/#�������������������������������������
#,/7/#
Y[ght{�����trhb[YYYYFHJTampzwurmaTRIHEDF()/6BDJOOOB6+)((((((ahjrt~����wth]aaaaaa���$'#�����15BNW[_a[NB5,-111111sz�������������zssss)/:O\u����h\OC6*#'&)������������������������������������������ ����������������������������&)+-+)&����rz��������}zwqrrrrrr%)5ABBNQSNB5)%%%%%%�����������������������������������������������������������������#%���������#0Ubn�����{b<0
�������������������st������������tphkss�����������������������������������������������������������������������������&).5;BNSNMB65)&&&&&&������������������������������������������������������������qtz������������tqqqq���������������������$*!��������������������������;<?AIUYabffb_WUI@<7;��������������������S[egntwtkge[YVSSSSSS���������������������������������������st|���������}tssssss������������������������������������������������������������o{�����������{tqoooo��������������������������	
���������� #<IUbhkhb[UI<0-#�� 

����������� 
$&#"
�����#*/40/# stz�������������yqpsRU[anyz����{zna\USPR������������������������

���������ֺ˺ɺ����������ɺֺ�����ݺֺֺֺ�ùñùù����������������ùùùùùùùù�[�Q�[�b�g�t�z�t�g�[�[�[�[�[�[�[�[�����w�~�����������������������������àÝÜàìõùùùìàààààààààà������������������������
��������������)�&�(�)�)�6�6�B�H�O�[�a�[�[�Q�O�B�;�6�)�H�H�D�B�D�H�L�S�U�Y�W�U�H�H�H�H�H�H�H�H�������������������������������������������������������������������������������˼ּ��������ͼܼ������#������������������������������
����!���
����M�L�A�A�@�G�M�Z�\�f�s�}������t�s�f�Z�M���������������ʼͼͼͼʼ¼��������������ݽؽн̽нݽ����ݽݽݽݽݽݽݽݽݽ��H�B�<�;�<�>�H�U�a�m�n�r�n�i�a�U�H�H�H�H�û������������������������ͻӻջһλĻ�����������������������������������������¿³²¦²¿��������������� ��������¿�z�t�o�k�tàÙÓÇ�z�u�wÇÓàìùþ��������ùìà�Y�S�M�@�4�2�,�4�@�M�P�Y�f�l�r�y�y�r�f�Y�ݽڽ۽ݽ���������������ݽݽݽݽݽ��<�;�/�/�*�/�8�<�H�U�U�W�U�L�H�=�<�<�<�<������ƳƮƲƳ�����������������������̻l�j�_�Z�]�_�l�w�x�z���}�x�s�l�l�l�l�l�l����������������������������������������ƳưƧƛƚƗƖƚƧƳƹ������������ƸƳƳƎƌƉƇƎƒƚƧƱƲƳƫƧƚƎƎƎƎƎƎ�;�8�/�.�/�6�;�E�H�T�T�T�N�I�H�?�;�;�;�;�ʾ������������ʾ׾��������	��׾�ƧƜƚƎƍƎƕƚƝƧƳƾ��������ƿƳƧƧ������������������$�(�0�2�2�0�+�$��ѿ̿Ŀÿ��Ŀѿݿ�������������ݿѿ��t�p�h�`�]�h�tāĈćā�y�t�t�t�t�t�t�t�t�f�Z�U�R�T�Z�_�f�s���������������s�h�f��������������������������������������������������������������������������{�z�v�{ǈǔǗǗǔǈ�{�{�{�{�{�{�{�{�{�{�6�5�6�:�C�O�\�\�]�\�O�C�6�6�6�6�6�6�6�6�������x�`�U�O�S�_�l�����û޻������л��y�u�q�z�������������ĿѿӿпĿ��������y�������h�_�Q�O�R�g�s���������������������Z�N�<�5�1�3�E�N�`�g�s�������������s�g�Z�s�q�h�k�s�{�������������������������s�s���������������������������������������������|�w�~�����������ɺֺ���ֺɺ���������ŹŰŭšŠŜŠŭŹ�������������������{�p�q�q�y�{ņŇňŊőŔŔřŔŇ�{�{�{�{���������������������������������������������������������������	��������������������������������������������������������t�n�g�m�s�����������������������������	�����������	��������	�	�	�	���׾ʾ������������ʾ׾������ �����Ç�z�n�i�j�o�tÇÓàãéìú����ùìàÇ���� ��
��#�0�<�>�I�I�I�I�D�=�0�#������ݽսؽݽ�������(�)�%�����3�(�,�3�<�D�J�L�Y�e�r�~���������r�Y�@�3�;�4�;�?�H�O�T�_�T�R�T�[�T�H�;�;�;�;�;�;�s�k�g�_�f�g�s�������������������s�s�s�s�.�%�$�.�3�;�G�T�`�m�y�{�y�v�m�`�T�G�;�.���	���	�����"�#�"����������	���	��"�"�*�(�"���������������$�/�$�!�����������V�O�I�=�2�=�A�I�V�b�o�p�t�o�i�b�V�V�V�V�'����� �'�4�@�A�@�?�@�E�@�4�'�'�'�'����������������)�2�6�?�6�1�)�����������������Ľݽ���߽߽��ݽнĽ��û����������������л��	�������ܻ��m�l�m�p�u�z���������}�z�m�m�m�m�m�m�m�mEVEPECE7E*E!E'E*ECEPE\EuE�E�E�E�E�E�EuEVE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E����
��
���#�#�/�<�<�H�J�H�C�<�/�#������������������ùϹܹ����ܹӹϹù����������������ùȹù¹�����������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� , X N ? L ; 3 � & : = C < M F  ; & 3 a S < a > 3 ; S D > s F ) 7 A ;  O N * 6 q 5 * 2 & ) : J ^ ^ % N : { A %  4 < g / g e Z x 3 2 1 Y 7 Z c U 4  O W      G  p  �    \  +  �  �  <  H  �  b  �  	  �    4  "  �  �    �  �  �  �  �      �    %  �    �  N  �  �  J  r  �  �  V  (  0  �  *  �  ,  �  �  �  �    �    �  L  �  r  �  �  �  �  �  �  �  !    �  `  !  7    `  Q  	  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  �  �  �  �  y  j  V  >    �  �  �  z  G  
  �  e    �  <  r  j  b  Z  R  J  B  3       �  �  �  �  �  �  �  �  �  �  :  5  1  -  *  &  "             �  �  �  �  �  �  �  �  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �    U  #  �  �  i  i  i  i  j  j  j  j  j  j  j  j  j  `  A  #    �  �  �  �  �  �  �  �  �  �  �  �  t  b  L  6    �  �  y  ?     �  �  �  �  �  �  �  �  �      �  �  �  4  �  �  C  �  �    4  ,  %                      �  �  �  �  o  J  &  �  �  �  �  �  �  ~  o  `  R  F  =  4  -  '            R  \  a  c  a  Z  J  2    �  �  �  �  `  8    �  �  o    -  f  �  �  �  �      �  �  �  w  .  �  W  �  �       �    $  %      �  %  -  /  .  *  #    �  �  z  M  $  �  �  �  �  �  �  �  �  �  �  �  �  |  _  A  0  7  #    �  �  z  K  b  p  t  r  n  f  V  C  &    �  �  �  �  b  C  &      �  �  �  �  �  �  �  �  �  �  �  t  d  R  ?  -         �  ,  [  �  �  �  �  �  �  �  �  �  �  �  |  R    �  �  ]  :  v  �  �  �  �  �  �  �  ^  V  �  �  �  |  /  �  K  �    �  �  �  �  �  u  f  W  H  :  -        �  �  �  �  �  g  6  �  4  E  ]  n  q  e  U  =    <  6  &      �  @  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  4    �  }  u  h  U  >  &    �  �  �  w  �  �  }  U    �  S  �  g  m  x  ~  x  l  [  H  3      �  �  �  �  a  @  !    +  g  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  m  f  _  X  Q  �  �  �  �  �  �  q  W  6    �  �    C    �  �  U  $   �  5  1  )      �  �  �  �  c  -  �  �  �  F    �  �  d    �  �  �  �  �  x  h  W  C  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  h  [  N  :       �  �  �  :  0  &    
  �  �  �  �  �  �  �  g  O  ?  V  [  L  ;  )  W  Q  K  B  7  ,      �  �  �  �  �  U  #  �  �  O     �  �  �  �  �  �  �  �  �  �  �  �  �  z  m  ]  N  >  .      M  T  W  S  H  4    �  �  �  �  u  Z  >     �  �  N  �  X        �  �  �  �  �  �  �  �  �  �  s  T  0  �  �  �  ]  �  �  �  �  �  �  �  �  n  <    �  �  ^  "  �  �  A  �  `  G  :  -    	  �  �  �  �  �  f  ?    �  �  }  @     �   �  i  d  `  [  W  Q  J  C  <  4  -  &  "                  #  e  �  �  �      (  +  *  !    �  �  Z  �  �  �  �   �  �  �  �  �  �  �  �  �  �  ~  e  M  4    �  �  �  �  �  �  n  �  �  �  �  �  �  z  ^  ?  !    �  �  �  `  -  �  �  �  }  �  �  �  �  �  �  �  s  e  N  .    �  �  �  V    �  �  M  @  3  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  6  z  �  �  �  �  k  '  �  �  \  �  �  ^  �  9  �  �    T  G  2  "    �  �  �  �  |  ]  >    �  �  �  >  �  �  <  �  �  �  �  b  6    �  �  L    �  �  c  3    �  �  B   �  �  �  �  �  �  ~  �  �  �  �  �  �  �  }  W  %  �  �  R    s  y  u  l  a  T  E  3    
  �  �  �  �  v  D     �  �  �  y  �  �  �  �  �  �  |  Z  1    �  �  z  N    �  �  /  �  	-  	�  
  
S  
�  
�  
�  
�  
�  
~  
4  	�  	s  	  �  �  )    o  l  R  A  )  	  �  �  �  f  3  �  �  �  D    �  P  �  �  �    Y  C  -  '  #      �  �  �  �  �  �  j  U  ?  %        �  �  �  �  s  `  K  4       �  �  �  �  d  D  #  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Z  .  �  �  |  -  �  H  �  �  z  e  Q  ;  &    �  �  �  �  �  �  �  �  s  `  N  <  4  0  '      �  �  �  �  ]  6    �  �  �  Y    �  u  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  d  F  �  �  i  �  �  �  |  ~  }  u  f  R  ;     �  �  �  z  G    �  �  f  �  �  �  �  �  �  y  t  l  _  M  6      �  �  �  �  �      )  @  H  S  Y  m  �  �    b  8    �  �  T    �  3  �  4  6  6  1  )         �  �  �  �  q  K      �  �  �  �  �  �      7  ?  8  .      �  �  X    �  [  �  .  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  J  B  9  /  !    �  �  �  �  \    �  �  F     �  �  C    �  �  �  �  �  �  t  Y  <    �  �  �  �  �  �  v  L  �  H  `  S  E  8  +  "        �  �  �  �  �  �  �  o  U  <  #  �  �  �  �  �  �  �  m  T  9    �  �  �  �  X  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  ^  C  %    �  �  �  s  F    �  �  n  B      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  b  �  �  �  �  �  [  3    �  �  �  �  ]  2     �  Y  �  1   �  Z  R  J  ?  0    �  �  �  �  X  '  �  �  �  D    �  l   �  �  �  �  �  �  �  �  j  O  7  #       �  �  �  �  �  �  Y  b  b  a  `  ]  P  B  4  (      
  �  �  �  �  �  �  �  o  
�  
y  
W  
)  	�  	�  	c  	  �  r    �    D  8  *  �  -  v  �  �  �  �  �  �  �  �  �  �  �  �  {  u  o  j  g  d  L  ,    N  �  �  �  �  �  �  �  e  7    �  �  P  
  �  |  =  �    �  �  *  Q  n  q  m  e  X  ?    �  �  B  �  r     {  �  �  7  (      �  �  �  �  �  v  S  1     �   �   �   �   \   6     '  <  '    �  !    �  �  }  7  �  n    �  -  �  R  �  4
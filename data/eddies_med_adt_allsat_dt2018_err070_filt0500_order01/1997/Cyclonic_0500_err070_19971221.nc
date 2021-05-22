CDF       
      obs    =   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��j~��#      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�r�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =e`B      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?���
=q   max       @E�          	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vzfffff     	�  *   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @��           �  4   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �hs   max       =<j      �  5   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�
�   max       B-E�      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��{   max       B-=�      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?�   max       C��K      �  7�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�i   max       C��5      �  8�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q      �  9�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  :�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =      �  ;�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M괢   max       P���      �  <�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�4�J�   max       ?�p:�~�       �  =�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       =e`B      �  >�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�33333   max       @E�          	�  ?�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vy\(�     	�  I   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           |  R�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̶        max       @���          �  S   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Dq   max         Dq      �  T   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�C,�zxm   max       ?�l�!-w       T�         Q                        1      !      >               C      +               /                  3                   =   #               
   
            2      	                  )   
      Nk�AN�Y�P�r�N�qdO�ND��N	�N���O�FN]��P�A N���O���N8K�Pz��Oy�CO�(�N:�cN�6<Pi7�N��oP@NfN�HORgO��O��O�d�OG�OgӄNֿ>O�>uN`�>O�,�O#cYNF�@N�&N� M�	�O�WPO%��N��N�g�N�!6O�yHN�0�N꠾N-cDN�mN(�*O�y�NTM�N�p�M��}O1>M��NZd�N�FO`�MN*�$O(k�N���=e`B<�9X%   �D���ě��t��t��#�
��o��j��j�ě�������`B���o�+�+�C��C��\)�t��t���P��P������w��w�,1�@��@��@��D���L�ͽL�ͽP�`�]/�e`B�y�#�}�}󶽉7L��7L��7L��O߽�hs��������������㽝�-���
���
���T���罰 Ž\�\�����`B#/<A@<3/%#enz�����ztnneeeeeeee��#<U�����{]I0���!)+-*)fp��������������tkdf��������������������Y[^hqmh[XRYYYYYYYYYY��������������������#/6<HLOHC</#��������������������Daz�������vtysUD��������������������mt���������������~tm�������������������=J[g�����������gB<5=)-5BNOMNHECDB5)% #)z���������������vtuz56<BMOOTOB:655555555��� ��������$5BN[h������g[KB3$$?BJKOU[hhhda[[ODB>??amz��������������laa��������������������CHIUUadnwxsna]UKHACC����������������������������������#/<H`kaan�znU</##���������������������������������������������

����������6O[glf^[YOB464fhptw�������thffffff
0<LU\`b`[UI<#
��������������������rtvy�����vtpjorrrrrr��������������������X[_hltu����th[TSRVXX���������������������
#,$!#-+&
�����#)//7<CHJJLJE</*#"!#lmz�}zqmjellllllllll��������������������nz��������������zpnn
"0<LU\aXZUI<#t{�������{xsqorrvt����

������������������~~����������bhnqnaXURTSUabbbbbbb����������������������������������������aanrz�������zxsnnaaa���������������������������������������������������������������������������������
 ��������������������������������������������)5BHNYNMF:5) rtt}������������ytrrŭũťŧŭŹž��������Źŭŭŭŭŭŭŭŭ����ÿÿ���������������������������������������g�Q�B�<�-�A�g����������������˺ɺ��ƺɺҺֺ��������ֺɺɺɺɺɺɻ����x�v�x�����������ûллû»�������������������������������������������������������ûлһڻлû��������������������H�F�?�<�6�<�H�U�`�a�c�i�a�`�a�d�a�`�U�H�6�.�)�$�'�)�,�3�6�B�I�O�Y�R�P�S�O�B�6�6�"���"�.�;�G�;�9�.�"�"�"�"�"�"�"�"�"�"��������������"�;�H�a�{�{�T�H�;��������ʾȾ������������������ʾξ׾����׾ʼY�R�P�M�L�B�N�Y�f���������������r�f�Yùòìäìîù��������úùùùùùùùù�z�g�[�a�t¦¿������������¿������������������"�/�;�C�R�N�;�"�����������ĿķįĮĳ�������
��*�)�-�������ؿ`�\�T�N�T�`�`�m�v�t�m�e�`�`�`�`�`�`�`�`����ھ�����	�	�����	��������ھž����ʾ����.�T�\�a�^�G�0�"�	����	����������������	��"�"�%�"�!��	�	��ƢƛƕƝƔƚƧƶ�����������"��������h�f�\�O�F�A�C�N�P�\�m�u�ƁƎƑƎƁ�u�h�#���
�
�	�
���#�/�6�8�<�D�<�9�/�#�#àß×ÓÊÂÀÀ�}ÇÓÖàæëïñìäà�B�6�������������B�O�[�f�c�m�g�^�B�A�6�2�3�A�Z�n��~���������������s�f�M�AŔŌŔŖŠŦŭŹ����������������ŹŭŠŔ�`�T�J�F�E�G�T�`�y�������������������m�`�������������������������������������������y����|�����������������������������s�q�g�^�Z�R�Y�Z�_�g�s�u�~�~�s�s�s�s�s�s�����������������������н��������н�čĈā�t�h�[�X�P�O�W�[�h�t�|āĉďĉĎč�����������������������������������ҺL�J�L�O�P�T�Y�e�r�t�~������~�r�e�Z�Y�L�������������'�3�1�-�-�'��������� �����'�(�*�-�(���������D�D�D�D�D�D�D�D�EE*ECEPETEUEPEJE*ED�D�E�E�E�E�E�E�E�E�E�E�E�FF$F-F+F$FFFE��(�&��(�5�;�A�D�A�5�(�(�(�(�(�(�(�(�(�(����������������������������������ÿĿпѿݿ��ݿۿԿѿͿĿ����F�=�3�-�,�/�:�S�_�x�����������������l�F���z�����������ûлջܻ߻ܻлû����������ѿ̿ͿͿѿؿݿ�����������ݿѿѿѿ��=�;�4�=�I�V�Y�X�V�I�=�=�=�=�=�=�=�=�=�=�f�]�Y�M�A�D�M�Y�Z�f�j�r�w�������s�r�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Y�M�L�Q�]�e�r���������������������~�r�Y�Z�X�N�B�N�X�Z�\�g�s�s�s�o�g�Z�Z�Z�Z�Z�Z�ܹ۹չϹɹϹйܹ�����������ܹܹ������������	�
���
��������������������<�7�#������#�0�2�<�>�I�U�U�V�U�I�<������������	��������������������������$�'�)�'�$�����������������������Ľн۽ݽ����ݽٽнĽ�����������ݻ��������&�/�0�-�'� ��������������ûŻͻû����������������������z�s�r�r�q�n�k�n�x�zÇàãâàÞÙÓÇ�z�U�N�H�<�9�8�<�C�H�U�a�n�q�n�m�e�a�W�U�U G ^ C G = 8 < R G H w � : . K j T ; . 1 2 / z / y \ K e K O i c  8 s N G � k J X Z | ' x + 3 - I 0 Q P � Q b M U " W ^ 5    �  �  =  �    h  8  �  I  �  P    F  O  !  G  �  L  �  �  �  0  N    �  �  �  �  �  �  �  �    z  �  0     v  &  �  &  �  C  	  ;    B  �  Z  �  Q    d  j  3  �    �  ;  �  =<j<�o���
��1��h�u��o��j�C���/��hs���m�h�C���^5�L�ͽixս�P�D�������㽡���'T���ixս����{�]/�u�aG����P�T���Ƨ��
�aG���%��7L�q����h�ě���+��O߽�����vɽ��㽟�w���P��Q콧�󶽧� Ž��Q콬1��E��ě��C���
=���#�hsB�Bk�B&��B��BJ�B\�BuHB!'B=�BgB��B ��B��B��B
��Bc$B+�BQ?BN�B	��B�/BB�NB�B�qB�RB��B}�B+�B�B��Bu B&�BlJB	�B!'3B��B�?B��B��A�
�B ��BpB%�RB(��B��B
�ZB�nB-;B��B��B?�B>mB'�B-E�B�OB�B��B��B��B
��B@LB�B&��B��B�hBM7BU,B đB)B��B2BB!?JB 5�B�BD(B@KB ��B��B?YB	�"B�UB ?uBơB@+B>sBD�B�1B�$B+9�B6GB��B��B%�B��B	�B!Q�B�5B��B��B?�A��{B*�B��B%ذB(�OB5�B
�B��B=�B~�B�BAB
�VB<�B-=�B�UBBB�B�B��B
��A��bA�`dA�Q@?�@���A��@��A�l�A���A`�[A��AO�@��EA�`.A��A�=�A�)�Ai��AX��A\^2A[F�B BO�A�� A��zA�(�A>�KA��Al�A�=1A�=dA���A%�}A��A��]?��?��A���C�m�C��KA�X�A�n�Az"�@���@��$A~6�B8�@�iC�u@�KA��-?�A�eCA���Af�B	V|A'α@�i�@���A�y�AŊ�A�w�AυjA�\/@D�@�|PA�t�@��A��A׈�AaΎA�}�AN�v@ޅ�A���A���A�F�A��Aj��AX��A[3A\�uB�B�qA��lA��A�R0A=�A�{@Ai �A��vA��A�OSA#�AۇA��?�A�?��?A���C���C��5A�}�A���Ax��@�u�@�5�A} BA/@ަ�C�	@��A��h?�iA�S�A�A�,B	9�A'�@��@��AɃ�A�8         Q                        1      !      >               D      ,               0                  3   !               =   $         	      
   
            2      
                  *                  A      !                  E            3               1      -            !   %            1      #                  #               %                                                            =                        9            )                     +            !               1      #                                                                                    Nk�ANO�P���N��O@1ND��N	�N���NĖN]��PSS�N���O���N8K�P1�Oy�CO�(�N:�cN�6<Of�N��oP:��N�HN���O��O��O;�ZOG�O4�	Nֿ>O�>uN`�>O�,�O�aM괢N�&N� M�	�Oa��Om�N��N�g�N�!6O��)N�0�N꠾N-cDN�mN(�*O��:NTM�N�p�M��}O1>M��NZd�N�FO-��N*�$O	��N���  �  8  �  9  "  �  �  �  �  �  b  W    0  �    �  �      P  �  �  	  h  �  �  �  �  ]  �  �  �    2  g  �  d  A  	      �  �  I  �  	  �  �  �  �  0  �  ,  �  +  �  [  �  �  	-=e`B<�1�ě���o�#�
�t��t��#�
���㼼j���ě�������`B�,1�o�+�+�C�����\)��P�t��'�P���Y���w�,1�,1�@��@��@��L�ͽP�`�L�ͽP�`�]/��hs����}�}󶽉7L��hs��7L��O߽�hs����������-���㽝�-���
���
���T���罰 Ž���\����`B#/<A@<3/%#oz�����zwpoooooooooo�#<n{����}XI<0���)+,))kv���������������tjk��������������������Y[^hqmh[XRYYYYYYYYYY��������������������#/0<HHKH@<//#��������������������n������ �����{x~zln��������������������mt���������������~tm�������������������FO[t������������NGCF)-5BNOMNHECDB5)% #)z���������������vtuz56<BMOOTOB:655555555��� ��������MNZgt���������tg[OMM?BJKOU[hhhda[[ODB>??bmz��������������lbb��������������������GHU^anqtnlaYUPHEGGGG����������������������������������!#+/<HUV]]WUKH</#!���������������������������������������������

����������6O[glf^[YOB464fhptw�������thffffff
0<LU\`b`[UI<#
��������������������stt�����trlssssssss��������������������X[_hltu����th[TSRVXX�����������������������
#$# 
 �����"##*/3;<=FHJIG@</+#"lmz�}zqmjellllllllll��������������������nz��������������zpnn
#0<GNSWPI:0#
t{�������{xsqorrvt����

������������������~~����������bhnqnaXURTSUabbbbbbb����������������������������������������aanrz�������zxsnnaaa���������������������������������������������������������������������������������
 ��������������������������������������������)5BHDB>85)$	rtt}������������ytrrŭũťŧŭŹž��������Źŭŭŭŭŭŭŭŭ���������������������������������������������g�U�F�<�?�L�g������������	��������ɺúǺɺӺֺ��������ֺɺɺɺɺɺɻ����{�x���������������ûƻû���������������������������������������������������������ûлһڻлû��������������������H�F�?�<�6�<�H�U�`�a�c�i�a�`�a�d�a�`�U�H�6�3�)�&�)�)�.�6�7�B�E�O�T�O�O�M�B�=�6�6�"���"�.�;�G�;�9�.�"�"�"�"�"�"�"�"�"�"���������"�/�H�T�j�u�w�t�_�H�;�	�����¾ʾȾ������������������ʾξ׾����׾ʼY�R�P�M�L�B�N�Y�f���������������r�f�Yùòìäìîù��������úùùùùùùùù¦�o�m�t¦¿��������������²¦������������������"�/�;�C�R�N�;�"�����������ĿķįĮĳ�������
��*�)�-�������ؿ`�\�T�N�T�`�`�m�v�t�m�e�`�`�`�`�`�`�`�`����ھ�����	�	�����	�������������������	��"�+�.�6�6�0�(��	���	����������������	��"�"�%�"�!��	�	��ƣƜƗƟƝƧƸ�����������!���������h�f�\�O�F�A�C�N�P�\�m�u�ƁƎƑƎƁ�u�h�#� ������#�-�/�5�<�@�<�4�/�#�#�#�#àß×ÓÊÂÀÀ�}ÇÓÖàæëïñìäà�B�6�������������B�O�[�f�c�m�g�^�B�M�K�A�=�8�9�@�A�M�Z�f�m�s�t�u�{�r�f�Z�MŔŌŔŖŠŦŭŹ����������������ŹŭŠŔ�`�T�L�H�G�T�`�m�y�����������������y�m�`�������������������������������������������y����|�����������������������������s�q�g�^�Z�R�Y�Z�_�g�s�u�~�~�s�s�s�s�s�s�����������������������н��������н��[�[�R�Q�Y�[�h�t�{āćčččćā�t�h�[�[������������������������������������ߺL�J�L�O�P�T�Y�e�r�t�~������~�r�e�Z�Y�L�������������'�3�1�-�-�'��������� �����'�(�*�-�(���������D�D�D�D�D�D�D�EEE*E7ECEOEPECE=E*EED�FF E�E�E�E�E�E�E�E�E�E�FFF$F)F)F$FF�(�&��(�5�;�A�D�A�5�(�(�(�(�(�(�(�(�(�(����������������������������������ÿĿпѿݿ��ݿۿԿѿͿĿ����_�F�:�1�/�3�:�F�_�x���������������x�l�_���z�����������ûлջܻ߻ܻлû����������ѿ̿ͿͿѿؿݿ�����������ݿѿѿѿ��=�;�4�=�I�V�Y�X�V�I�=�=�=�=�=�=�=�=�=�=�f�]�Y�M�A�D�M�Y�Z�f�j�r�w�������s�r�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��Z�O�N�R�^�e�r���������������������~�r�Z�Z�X�N�B�N�X�Z�\�g�s�s�s�o�g�Z�Z�Z�Z�Z�Z�ܹ۹չϹɹϹйܹ�����������ܹܹ������������	�
���
��������������������<�7�#������#�0�2�<�>�I�U�U�V�U�I�<������������	��������������������������$�'�)�'�$�����������������������Ľн۽ݽ����ݽٽнĽ�������	��������������!�'�+�-�*�'�������������ûŻͻû���������������������Ç�{�z�t�s�t�u�zÇÊÓàâáàÝØÓÇÇ�U�N�H�<�9�8�<�C�H�U�a�n�q�n�m�e�a�W�U�U G A D F 7 8 < R G H _ � : . H j T ; . * 2 0 z P y \ ) e B O i c  # f N G � l J X Z |  x + 3 - I , Q P � Q b M U  W F 5    �  s  �  �  �  h  8  �  �  �      F  O  "  G  �  L  �  �  �    N  �  �  �  �  �  |  �  �  �    2  G  0     v    I  &  �  C  Q  ;    B  �  Z  p  Q    d  j  3  �    n  ;  N    Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  Dq  �  �    p  a  P  <  %  
  �  �  �  t  M  %  �  �  �  �  V  '  ,  2  7  4  .  )  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  i    �  �  X    �  �  �  A  �  ^  �    '  -  8  8  3  -  &  2  (    �  �  i  .  �  �  U  �  �    �  �         !            �  �  �  �  Y  #  �  �  �  P  �  �  �  �  �  �  �  �  �  �  �  �  ~  x  t  p  e  9    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  &  �  �  �  �  �  }  |  �  �  �  g  B    �  �  �  f  ?    �  c  d  f  �  �  y  ]  @    �  �  �  �  [  8    �  �  �    �  �  �  �  �  �  �  �  �  }  t  i  ^  S  H  9  (      �  N  [  [  Z  Z  A    "    �  �  q  M    �  H  �  }  �   �  W  T  Q  N  L  K  I  H  F  E  0    �  �  �    `  >     �    �  �  �  �  �  n  ?    �  �  g  4  �  �  }    �  K  �  0  $      �  �  �  �  �  �  z  d  O  :  '       �  �  �  �  3  i  �  �  f  6  �  �  �  �  �    �  �  )  o  �  �      �  �  �  �  �  �  t  Q  ,    �  �  H  	  �  �  �  U  �  �  �  ~  ^  C  '  	  �  �  �  P    �  �  ?  (  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  ~  `  7    �  w  -  �  �  D  �  �    G  �  �  �  �  �  �  �  �  �  �       �  �  K  �      �  P  K  G  B  =  8  3  +  !         �   �   �   �   �   �   �   �  �  �  �  ^    �  �  h  #  �  �  A  �  �    �  e  A  �  c  �  �  �  �  �  �  �  �  �  t  `  L  =  2  &            �  �  �    	  	  	          �  �  �  �  �  ~  W  	  �  h  L  ,    �  �  �  �  �  Q  "  �  �  �  `    �  �  Y    �  }  j  W  a  _  =    �  �  g    �  :  �  A  9  Z  �  �  �  !  �  �  �  �  �  �  �  �  V    �  S  �  D  �  �  <  �  �  �  �  �  �  x  i  Y  H  6    �  �  c  �  �  X    �  ~  �  �  �  �  �  �  �  �  �  �  �  m  =    �  �  �  r  �  v  ]  \  Y  T  Q  M  E  ,    �  �  �  �  m  H  #  �  �  �  �  �  �  a  +  �  �  �  �  �  �  Z    �  �  �  �  ^    �  t  �  |  s  k  a  T  G  9  +      �  �  �  �  �  �  �  �  w  �  �  �  �  �  �  p  V  @  -    �  �  |  '  �  K  �  �  �  �      �  �  �  �  �  �  b    �  �  L    �  �  Q  �  �     %  *  .  3  7  <  A  ?  3  '        �  �  �  �  �  �  g  @  4  9    �  �  �  �  x  Z  :    �  �  �  ]  !  �  �  �  �  �  �  �  �  �  �  �  �  s  ^  A    �  �  z  �  �  1  d  a  ]  Z  Q  .    �  �  �  u  N  &  �  �  �  �  ]  4    
�  
�  
�    :  ?  6    
�  
  	K  w  �  �  O  �  #  J  1  �  �  �  	  �  �  �  �  E  �  �  X    �  9  �    l  �  �  �      �  �  �  �  �  �  �  }  j  U  @  +    �  �  �  y  Q    �  �  �  �  �  �  �  �  k  P  6    �  �  �  �  �  �  w  �  �  k  M  0    �  �  �  �  �  �  �  �  �  �  @  �  H   �  |  �  �  �  �  �    m  ]  K  <  7  )    �  �  �  �  L  
  I  C  <  4  ,  .  3  ,         �  �  �  �  �  z  `  E  *  �  y  i  Z  K  6       �  �  �  r  W  :    �  �  �  b  1  	       �  �  �  �  �  �  �  �  �  �  �  �  �  ~  g  Q  ;  �  �  �  �  �  r  V  8    �  �  �  M    �  �  W  �  �  �  �  �  �  �  �  �  �  �  �  �  q  \  G  2    �  r  :  
  �  �  �    c  =  	  �  �  @  �  �  p  -  �  �    �  2  �  ~  �  �  �  �  �  �  �  �  {  `  F  +    �  �  �  �  �  �  �  0  /  .  .  /  '        �  �  �  �  �  �  n  v  �  �  �  �  �  �  �  �  �  �      	           '  -  3  9  @  F  ,         �  �  �  �  �  �  �  v  _  ?    �  �  �  I    �  ~  z  w  s  o  k  b  W  K  ?  4  (        �  �  �  �  +         �  �  �  �  �  �  p  K  &  �  �  �  f     �   �  �  �  �  �  �  �  �  �  �  t  c  N  1    �  �  m  G  %      P  U  [  R  C  -    �  �  h  $  �  �  :  �  D  m  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     I  q  l  p  �  �  ~  `  @    �  �  �  f  .  �  �  e    �  [  �  	-  	(  	  	  �  �  �  f  /  �  �  l  %  �  >  �  4  �    *
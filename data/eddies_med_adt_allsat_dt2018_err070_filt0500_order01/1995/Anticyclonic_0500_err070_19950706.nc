CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?����+      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�,>   max       P��2      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\)   max       =��      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @E�
=p��     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vo\(�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q            t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >=p�      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B.�r      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��R   max       B.�       �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?;   max       C�c�      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��   max       C�h      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          =      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          5      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�,>   max       P��      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���3��   max       ?�T`�d��      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\)   max       =��      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @E�
=p��     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min                  max       @vo\(�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���vȴ:   max       ?�T`�d��     �  QX      	                  �            @            	   -      &   @               
                4      (   O   �   <                        .      #   !      
   	             H         3   OU�NYQ�O���N�B�N�}%N�<N�O�PlY\N�:sN��N/P��2NL]|O,
N�j�N�xO��'NK��PA��POx�O�ǖOs@OZ4M��OPX�O�kOǶWNJ�qN��PO��OD$O���P�TdO�9wPX�N.��O��M�$�N��@M���NQTsOOuRO�m�M�,>O�]�O�V<N�N׀�O��N�4Or5"N*N�O��N�zO!��O���NkYB�\)��󶼓t��D���D���ě���o%   %   %   ;D��;�`B<o<#�
<49X<T��<e`B<u<�C�<�C�<�t�<���<��
<��
<��
<��
<��
<��
<�1<�9X<�9X<���<���<�h<�<�<��=C�=t�=�w=�w=8Q�=D��=H�9=H�9=L��=Y�=]/=]/=u=��=��=��P=� �=�v�=�
==��������������� �������������������������INXt�����������gZQNIlinz�����ztnllllllll�������������������� #//14550/#��������������������1027BNg��������g[NB1�����

����������BBEO[fhqlh[OHBBBBBBB\[`alnz�zqna\\\\\\\\����)BtzgSC5)�����
),,)����������������������������������������"!��������������������"&,/0/)"wz����������������w�����
/9DKMH<#
������������

����  #/<CHKQRHF</+&%#  ����'(%$��)*-)025BEN[gqt{}tqg[NB50����� $-2251���316BO[chopyzqh[XFB73��������������������AENP[gt���zuha[SNGBA������������������������������������������������	������������&21,���������������

������&<BIY]`_\NB)��������������������������������������������������������������'"&)5BDNBB5)''''''''`admzz{zma`````````` �)        ���������������!)/;>HTaed^TH;/,-%%!nz}}zqnknnnnnnnnnnn�������#�������������������������SLIU\ahdaUSSSSSSSSSSstz�������������|xts��������������������25BNYTNB?52222222222����������������������������������������������������������������")6BIIFB76/)$)6<?A@>63)"!#/<<><:/#""""""""�:�G�S�`�l�y���y�n�l�`�Z�S�G�:�.�-�.�9�:�$�0�=�?�=�5�0�$�����$�$�$�$�$�$�$�$�����������������������������������������������������������������������������������������������������z�x�y�z������������D�EEEE'E*E7EAE7E7E*EEED�D�D�D�D�D��\�a�`�\�[�O�C�;�6�2�-�4�6�C�O�\�\�\�\�\�)�B�[�tĉđĐĆ�j�[�B�6�����������)�B�F�I�B�?�6�.�)�%�����)�1�6�=�A�B�B����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E���I�b�{œŝŉ�s�b�<�
������������������M�Z�_�d�]�Z�M�A�C�L�M�M�M�M�M�M�M�M�M�M��"�.�7�5�.�*�"��	������������	��¿����������������¿¸²¦¦¬²¹¿��(�*�)�(���
�������������5�A�N�Z�e�f�e�`�Z�N�A�5�(������(�5�/�3�;�D�;�/�'�"�����"�-�/�/�/�/�/�/�����	��%�'�&�	�������������������������G�`�m�����������y�`�G�.����� �	�?�G�A�M�Z�f�����������k�M�A�4�-�'�+�2�4�A�������	�������������������������������������� ����������������������������	���	�����������������.�;�C�R�T�a�g�k�`�_�T�G�=�;�7�-�(��#�.�	�"�.�;�N�T�U�M�G�;�.�"��	��������	�ʾ�������׾������s�f�Z�S�U�f������ʼ������ʼ˼ʼ¼���������������������������������,�0�(�$������������������=�hāā�[�H�)�������ðéð�����뻑�������ûȻ̻û��������������~�����������*�6�?�N�Z�`�\�M�6��������������������"�4�9�>�=�/����������|����������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDqDuDwD{�������0�=�H�0��������ƤƔƄƂƌƧ���(�0�5�8�5�.�(������(�(�(�(�(�(�(�(�������(�4�A�C�I�B�A�4�0�(�����������������������������/�;�H�J�H�H�H�;�/�/�#�-�/�/�/�/�/�/�/�/�ĿǿѿԿѿѿĿ¿��¿ĿĿĿĿĿĿĿĿĿĿ����������������������������������������y�����������~�y�l�`�S�P�N�M�N�S�`�l�u�y�0�I�P�Q�I�A�1��
��������ĺĶĿ�����
�0��������������������������ʼּۼ������ּ˼��������������������(�4�A�M�Z�^�_�b�\�M�4�(���������s�������������s�p�r�s�s�s�s�s�s�s�s�s�s���"�%�"����	�����������������	���������������������������������������ĽȽ̽ɽĽ����������ĽĽĽĽĽĽĽĽĽ�¿�������
��!�����������¸¿¿»»¿�n�w�w�q�n�a�U�V�a�l�n�n�n�n�n�n�n�n�n�n�ܹ���
����������ù��������ùϹܺ����������������ֺɺȺɺպ⻷�ûлܻ����������ܻлͻû��������e�~���������������������~�r�Z�Y�M�O�W�e��*�6�=�B�6�6�*�!����������� U A : B G F * 2 Z = F 3 : D \ J , V G " < 4 o H W 2 u 8 � M N 2 +  D 5 1 S D ; e H W A J X K U C n Q V 1 V c /     O  `  �  �    0    �  �  �  b  >  \  5    <  �  ]  �  c  3  7    "  �  �  r  o  �  �  �  �  �  �  �  L  B  &  �  (  |  �  �    /  1  )  '  "  i  (  [  �  2  �    s���
��j;D���ě����
<#�
;�o>=p�;�o;ě�<o=�\)<T��<u<��<�1=m�h<��
=]/=��
=8Q�=,1=\)<�j<�h=0 �=T��<���<�/=��P=49X=�o=���>8Q�=�Q�=C�=49X=�P=#�
='�=49X=y�#=��=Y�=�1=��=m�h=�%=�%=�%=�E�=Ƨ�>z�=��=�l�>�R>VB�aB�B
Ba�B˱B�PB �B	E�B�pB�B��B�\BJB �B�B�UB�rA��B�B�B#�KB�fB�Bk�B�PB�B�ZB"��B	<B"B!�1BcB�}B�VB�B��B�vB �5BJgA��]B.�rB-%WA�MsB�B�#B6�B�eB
�9B�B�`B@�B��BB��BxB�B0�B�XB�9B	�lB�ZBfB��BżB	?�B@B��B�B��B@YB��B��B?CB�A��RB?�BD�B#�~B�^B��BD�Bi�B?ZB�/B"��BL�B��B!}�BL�B�B�.B�UB��BAB ��BeA���B.� B-GFA�~�B8lB��B?�B,�B
��B�B�VB��B�B�#B��B��B B>vA�B	�TA�c A��,A��C�nB _A�KyA���At��C�c�A�6A>NA\�[A���A4ٹA���A�xA�F Ae�YA>��Aҕ�A��]AYoAdTA]�AL�u@�V1A�wAӠ[@��A���A��QC��dB��A�CVA6��A1��A�l�Ay�`As��A�<A��@�x5@��:A:�zA���A���A���A&��A���A���?;@Nn�@�d�@��A��iA�B	�,A��"A�}�A�q�C�g\B �XA�{dA�o�AuLC�hA��A>��A\>�A�� A3#�A���A�Q,A�D@Ah�tA?<(A�{�A�m;AYEAb��A_ 6ALԑ@�rA�vA�o_@��uA�v�A��jC���B=�A�{aA5VMA2�A�pAy�	At��A��A撕@�Go@��MA;J�A�{A���A� �A'�A���Aƀu>��@S!/@�P�@,�A���      	               	   �            @            	   -      &   @               
      !         5      (   O   �   =                        /      $   !      
   
         !   H         3                           3            =                     1   /   !               #   '         9         ;      1                        %   
                                                                                                %   '                              +         5      #                           
                                       OU�NYQ�OWD;N�B�N�}%N�<N�O�O`XFN�:sN��N/O�K�NL]|O,
N��N�xO,�NK��P"P��O��O��O?�M��OPX�O��nORv\NJ�qN��P2��OD$Op{�P��N�i(OވrN.��O��M�$�N��@M���NQTsOOuROs�mM�,>ORN�,bN�N��QO��N�4Od�N*N�O�M�N�zO!��O���NkYB  .  �    S  b  �    �  �  `  p      �  �  �    �      �  �  o  0  �  N  2  �  �  s  �  ~    �  �  ~  �  �  �  <  �  !  %  �  �  G  �  �  �    1  I      h  	8  ~�\)���T���D���D���ě���o=��%   %   ;D��=@�<o<#�
<u<T��<�<u<�9X<�`B<�/<��
<�1<��
<��
<�1<�<��
<�1<�/<�9X=C�<��=�h=L��<�<��=C�=t�=�w=�w=8Q�=y�#=H�9=]/=}�=Y�=aG�=]/=u=�+=��=� �=� �=�v�=�
==��������������� �������������������������UT[gt����������tg`[Ulinz�����ztnllllllll�������������������� #//14550/#��������������������MIIJNP[gtz�����tg[NM�����

����������BBEO[fhqlh[OHBBBBBBB\[`alnz�zqna\\\\\\\\�����#+752)��
),,)����������������������������������������"!��������������������"&,/0/)"���������������������������
/<AGH/#�����������

��!!#/<BHKPQHE</,'&#!! �� %'$# )*-)025BEN[gqt{}tqg[NB50�����#,113)���<:9:?BOS[_`ehg[OMGB<��������������������AENP[gt���zuha[SNGBA������������������������������������������������	
�����������)01+�����������������������)5BINUUO?5)������������������������������������������������������������'"&)5BDNBB5)''''''''`admzz{zma`````````` �)        ���������������.1;HRT[_aa^TH;510//.nz}}zqnknnnnnnnnnnn���������������������������������SLIU\ahdaUSSSSSSSSSS}yu}����������}}}}}}��������������������25BNYTNB?52222222222����������������������������������������������������������������")6BIIFB76/)$)6<?A@>63)"!#/<<><:/#""""""""�:�G�S�`�l�y���y�n�l�`�Z�S�G�:�.�-�.�9�:�$�0�=�?�=�5�0�$�����$�$�$�$�$�$�$�$�����������������������������������������������������������������������������������������������������z�x�y�z������������D�EEEE'E*E7EAE7E7E*EEED�D�D�D�D�D��\�a�`�\�[�O�C�;�6�2�-�4�6�C�O�\�\�\�\�\�6�B�O�[�e�h�q�q�l�h�a�[�O�B�6�.�(�)�4�6�B�F�I�B�?�6�.�)�%�����)�1�6�=�A�B�B����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��#�0�<�I�R�\�\�U�S�I�<�0�#���	����#�M�Z�_�d�]�Z�M�A�C�L�M�M�M�M�M�M�M�M�M�M��"�.�7�5�.�*�"��	������������	��¿��������������¿²¦¢¦°²¼¿¿¿¿��(�*�)�(���
�������������(�5�A�N�O�Z�Z�\�Z�Y�N�A�5�(�#�����(�/�3�;�D�;�/�'�"�����"�-�/�/�/�/�/�/�������	�� �!��	�����������������������;�G�T�`�c�m���������l�`�G�.������;�A�M�Z�f�s�v�����}�s�f�[�M�A�<�4�2�6�A��������������������������������������������������������������������������	���	�����������������.�;�C�R�T�a�g�k�`�_�T�G�=�;�7�-�(��#�.�	�"�.�;�L�S�T�K�G�;�.�"��	��������	�����ʾ׾����ؾʾ�������x�}��������������ʼ˼ʼ¼���������������������������������,�0�(�$������������������9�T�[�R�D�)�������ôîù�������뻑�������ûȻ̻û��������������~�����������*�/�6�B�L�K�C�6�*������������������"�0�6�:�9�4�$���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ƧƳ������� ����� ������ƧƞƚƚơƧ�(�0�5�8�5�.�(������(�(�(�(�(�(�(�(�������(�4�A�C�I�B�A�4�0�(�����������������������������/�;�H�J�H�H�H�;�/�/�#�-�/�/�/�/�/�/�/�/�ĿǿѿԿѿѿĿ¿��¿ĿĿĿĿĿĿĿĿĿĿ����������������������������������������y�����������~�y�l�`�S�P�N�M�N�S�`�l�u�y��#�1�/�#�"��
���������������������
�����������������������������ʼּټ�����ּʼ¼��������������(�4�A�M�V�Z�Z�\�Z�M�A�4�0�(��'�(�(�(�(�s�������������s�p�r�s�s�s�s�s�s�s�s�s�s���	��"�$�"����	���������������������������������������������������������ĽȽ̽ɽĽ����������ĽĽĽĽĽĽĽĽĽ�¿���������
�� ���������������¼¼¿�n�w�w�q�n�a�U�V�a�l�n�n�n�n�n�n�n�n�n�n�ùϹܹ�����������ܹϹù��������ú����������������ֺɺȺɺպ⻷�ûлܻ����������ܻлͻû��������e�~���������������������~�r�Z�Y�M�O�W�e��*�6�=�B�6�6�*�!����������� U A < B G F *  Z = F ( : D j J ! V ?  3 3 j H W . f 8 � 6 N " %  9 5 1 S D ; e H D A D + K H C n T V 0 V c /     O  `  �  �    0    �  �  �  b    \  5  �  <  j  ]  �  �  R  (  �  "  �  l  �  o  �  #  �  �  b  �    L  B  &  �  (  |  �  �    �  �  )  �  "  i    [    2  �    s  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  .  ,  &                �  �  �  �  �  �  N    �  �  �  �  �  �  �  �  �  �  �  �  t  ^  L  N  Q  >  $  	  �  �  �  �  �  �      �  �  �  �  �  �  �  �  P  3  &    �  +  S  P  L  H  D  ?  :  3  *  !          �  �  �  �  �  �  b  b  b  _  X  R  H  =  2  '         �  �  �  �  �    V  �  �  �  �  �  �  �  v  a  J  /    �  �  �  �  b  >    �        �  �  �  �  �    e  H  )    �  �  �  �  �  �  �  
*  �  �  �  �  �  W  �    ]  �    E  �    �  �  
�  [  �  �  �  �  �  U  4  ,  #      �  �  �  �  z  m  f  _  X  R  `  _  _  ^  Y  T  N  H  @  9  /  "      �  �  �  �  �  }  p  i  b  \  S  F  8  +    
  �  �  �  �  �  �  �  �  �  �  
  �  �  �  �  �  o  �  �  �            �  �  �  ;  '            �  �  �  �  �  �  �  �  �  x  c  M  5      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  ]  >     �  H  z  �  �  �  �  �  �  �  j  ?    �  �  }  I  '  �  .  �  �  �  �  �  �  �  �  �    C  U  M  E  8  +      �  �  �  �  ?    �  �        
  �  �  �  O    �  b  �  �    5  �  �  �  �  �  �  �  i  N  1    �  �  �  �  {  Z  :     �  �  �  
    �  �  �  �  �  ~  X    �  �  \    �  f    �  �  �        �  �  �  U    �  �  4  [  '  �  �  �    j    Y  }  �  �  �  �  �  �  �  w  W  2  	  �  �  p  7  �  �  �  �  �  �  �  �  �  h  G  #  �  �  �  p  0  �  v    �  7  k  n  i  Y  A  &    �  �  �  ~  p  d  a  p  V  )  �  �    0  ,  (  $                 �  �  �  �            �  �  �  �  �  �  �  t  c  V  K  J  W  \  P  A  $    �  �  F  N  M  J  F  ?  6  -         �  �  �  b  1    �  �   �  �  �           2  ,    �  �  �  �  G  �  �    c  �   �  �  �  �  �  �  �  �  v  e  N  8  !    �  �  �  �  �  �  �  �  �  }  n  l  l  m  _  J  5  %        �  �  �  �  �  �    6  k  8    �  �  �  \  &  �  �  ]  �  x  �  �    m  !  �  �  �  �  �  }  g  K  '  �  �  �  o  A    �  �  L    �  h  v  x  |  ~  ~  z  k  U  8    �  �  d    �  C  �    f  �      �  �  �  �  _    �  Y  �  �  8  �  N  �  �  �  '    7  �  �    |  �    U  �  �  �  �  c  �  &  `  ~  �  �  A  �  �  �  �  �  �  �  �  �  �  �  �  �  f    �  �  (    ~  t  j  `  W  K  <  -      �  �  �  �  d  @     �   �   �  �  �  �  �  �  �  z  j  Z  H  1    �  �  �  �  e  9  )  =  �  �  �  �  �  �  �  �  �  �  |  n  `  S  I  ?  5  +  !    �  �  �  �  �  �  �  o  [  G  5  %      �  �  �  �  �  �  <  0  $        �  �  �  �  �  �  �  �  �  {  l  ]  O  @  �  �  �  �  �  �  v  e  W  L  B  8  &     �   �   �   �   �   �  !      �  �  �  �  �  a  -  �  �  d  $    �  �  �  }  )  �  V  �  �    !  $      �  �  �  �  m    {  �  �  �  �  �  �  �  �  �  �  �  �  �  t  i  `  V  M  D  7  )      �  W  J  ?  �    s  j  U  8  
  �  �  7  �  �    �  )  �  +  u  �  �      .  >  F  :  "  �  �  �  ?  �  �    �  1  ~  �  �  �  �  �  �  �    r  c  S  D  5  '        (  :  K  {  �  �  �  �  �  �  u  e  S  ?  (    �  �  �  �  �  {  ]  �  �  �  �  q  _  N  :  $    �  �  �  �  �  k  M  /    �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    .      �  �  [    �  �  }  T  %  �  �  �  �  �  u  �  I  �    k  �  �    '  �  �  Y  )    +  U  ~  �  �  �     �  �  �      �  �  �  �  B  
�  
�  
&  	|  �  �  �  �  �  �      �  �  �  �  �  y  X  7    �  �  �  �  `  =    �  W  h  Z  G  '  �  �  �  p  3  �  �  c    �  a    �  �  D    	8  	2  	/  	-  	/  	5  	7  	5  	)  	  �  �  ?  �  0  �  �  �      ~  ]  ;    �  �  Q    �  �  V  �  �  N  �  �    �  ,  �
CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�1&�x��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nj�   max       P�x�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =�h      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E���
=q     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��33334    max       @vt��
=p     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @N�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @��`          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �#�
   max       >Y�      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B-�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��e   max       B-F�      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��D   max       C��'      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C���      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nj�   max       P;�C      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��,<���   max       ?�?���      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       =��#      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @E���
=q     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vt��
=p     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @         max       @N�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Α        max       @�)�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?�<�쿱\     0  O�      
                                 �      	   $   /            6   	               T      .      ?      Z      	      .   #   +      -               �                  %      N   N�@�N3�Oy�}O�X9N�k�N�g�OK��N*��N���Ou;�N���N���P�,�Ou�O!�/O���P�x�Nj�NghIO�X�O��NE��N��O�'N�)�N�J�P)�O��nO�h�N��fP6�O�P8�:N�e�Ng��O

*O�R�PߢO��O JzO0��N��N���N��@O��O��Nچ�NT��N��UN �<O$�XOz��N�DOTpsN�����j����u�e`B�D���#�
��o��o%   :�o:�o;�`B<D��<e`B<�o<�t�<���<�1<�j<�/<�/<�/<�/<�`B<�<��=C�=C�=\)=�P=��=��=��=�w=,1=,1=L��=T��=m�h=�o=�o=�o=��=�C�=�C�=�O�=�\)=�\)=�t�=��P=���=�9X=�v�=�/=�h�����	 $�����][Zanrqnma]]]]]]]]]]`\[_ckpty��������tg`[Z[]gt����������tgb[�����������
(($"��������������������RQSTabggaTRRRRRRRRRR��������������������������������������������
#+*##
�������"/;DD=;8/"01<=65BNY������g[I:0� 
#(0/0-$# 
TORVW[ehw�������th[T��,O[ae`_[OB6)��� )5NUWde_TNN5��$),)����������������������������������������������������������������������������������������������

������������������������a\behjtw}~���|ttkhaa������)4HGB5)��vvz��������������{v���#/5<A@<8/#
��������������������������)5N[^ee[N=)������� 

���������#(5@DNNJ<��rtz����������zrrrrrr;02<HQUYUH?<;;;;;;;;50/18<HJU\aia`UPH<55kz������������|yztsk)BNew�zgN5)��);BN]eOH:6)�������������������������������}x��������������}}}}���������������������������������������������������

������63;;DHT^aca]VTH;6666��������������������hhmst��������}thhhhh����������������������������������������IGA<#		#/4=BI,*/2<HOTPH</,,,,,,,,gcbenz����������zqng�������������������ܼ��ʼּ����߼ؼּʼ�����������������E�E�E�F	FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E��a�n�zÓàìù��ùõìàÓÇ�z�m�c�Y�Z�a�����������������������������������������F�S�_�l�x�}���x�l�_�V�S�P�F�:�@�F�F�F�F��"�%�.�:�8�.�"��	�����	����������������������������������������������������������������������������������������5�6�A�B�B�A�5�5�(�!��#�(�4�5�5�5�5�5�5������4�0�(������������ۿڿݿ��������������������������������a�f�m�r�y�x�m�a�T�S�T�U�W�X�a�a�a�a�a�a�)�B�tāĚĦĮĭĦďă�h�O�)����������)�m�y���������������y�l�`�T�G�A�G�O�T�^�m�A�M�Z�f�s��s�q�f�[�Z�Q�M�A�@�6�:�5�5�A���׾��������׾���������������������������(�C�X�f�Z�5����ݿĿ������{�~�����Z�f�m�j�f�a�Z�V�T�X�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������������Ҿf�s�������������������s�f�Z�T�T�V�Z�f���Ϲܹ����� �����Ϲ������������������#�/�0�;�8�/�#��� �#�#�#�#�#�#�#�#�#�#���¼Ƽʼͼּ׼ּԼʼ���������������������)�5�B�I�N�O�N�B�=�5�)��������ù��������������ùìàÓÒÓàæìöùù�-�:�F�S�U�S�J�F�:�-�(�!���
��!�$�-�-�
�#�<�M�]�b�l�_�P�<�0�
���������������
���(�5�N�V�b�g�l�a�Z�A�$�������������������������������������v�o�p�w�����������������������������Ƨ������������������ƳƚƄ�z�u�j�p�{ƐƧ��(�.�4�A�G�M�Y�X�M�A�4�(����������׾����������ʾ������z�_��������B�O�U�[�`�[�[�O�B�6�2�3�6�<�B�B�B�B�B�Bìù����������ùôíìììììììììì���������������������������������������������ŹŭŠŞŦŤţŜŠŭŹ�������	��;�H�a�g�`�Z�;�/�"��	�������������	�e�����ʺ̺ͺպֺۺɺ����������|�h�`�a�e�4�@�M�Q�Y�f�l�m�f�Y�V�M�I�@�?�;�8�4�1�4�@�M�R�Y�\�\�Z�Y�M�@�4�'�%��"�'�-�4�;�@����������������������������������������¦²¿����������¿²¦£¦¦¦¦¦¦�	��"�#�(�)�%�"��	������������	�	�	�	�������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D{DoDiDkDpD{D�D��U�b�n�n�{�~��{�x�n�h�b�U�S�I�K�U�U�U�U�����ĽнܽнǽĽ������������������������4�4�@�M�Q�X�M�G�@�4�0�'�&�'�)�2�4�4�4�4�������������������������n�zÇÓàâåäáàÓÇÁ�z�s�n�m�k�h�n���������������*�6�C�O�V�N�<�6�*��EEEEE"E#EEED�D�EEEEEEEEE�ܻ������������ܻлƻû��ǻлӻܻS�_�l�x���������x�l�_�S�F�E�F�G�S�S�S�S 6 G 4 2 H 8 J H E 7 X U ; ? F V U _ g - 2 C z , n W D `  f  D G * A . W 9 H B = O % 4  + < > 4 k < G 4 * m    '  C  
  a  �  �  �  9  �  �  �  �  �  z  �    �  A  �  	  �  q  �  J  �  �  '  k  �    �  4  1  �  v  0  �  �  $  D  �  %  �     �  �  �  p  �  a  p    �  �  ��#�
�#�
<e`B<t�:�o��o<o;D��;��
<�C�;ě�<e`B>L��=t�<���=]/=�+<ě�<�h=49X=��T=\)=C�=,1=<j=t�==m�h=��
=@�=���=L��>J=L��=P�`=�7L=\=� �=���=��=�/=�1=��=��=��>Y�=��=���=�^5=���=�S�>   =��>>v�=���B�!Bl�B	�B	߁B�PBXBpbA�'�B��B�TB��A��B	y�BV!B��BUrB�OB*^B>?B��B�BS�B"G�Bm�B!ʿB*B�B�NBc�B"MB�B��B��B J�BBB0B��B�ABqB�B\�B�B�qB�BMYB�A���B+<�B��B�uB0)B��B؍B�B-�B��BC�B
H�B	��B��B�B��A���B��B@�BA��eB	E�B?�B�<B��B��BA�B�iB�^B��B~�B"��B:�B"@[B�;B�JBHcBy�B"?�B{B� B?�B DB�B=zB�B�
B�B;/B�sBCYB�gB �B��B@iA�}VB+AKB�&BĔB@B�B��B��B-F�@��cC��'A�u>A���@��A^.A��3A�N�A��A��~A���A���A�L�Ak�,A>
SAPU�A}CZA?�uA�IwADJ�>��DA��@��A��A�M@v��A���A�''A�/�@��BnA8�APE�A��A��XA�,�A���A��4@��@��,@҄s@KA��MA�#�A�(�C�ЊA��A&-G@��tA0JnAɋA�ǔC�dh@���@�M�@��mC���A�A�ye@��JA]��A�@�A�_?A�TA���A�v�A�kA�|�AkZkA=3�AO�A~��A?�AЛ�AC�n>���AC@�QA�qKA��@z}�A�J�A�dKA�~@���BXA9�AS��A�h;A͐kA��A�lA�@�b@���@���@ A�[�A��)A��]C��A��^A%]�@�_0A0�AɆA��C�hE@���@���      
                                 �      
   %   0            7   	               T      .      ?      Z      
      .   #   +      -               �                  &      O                                          9         %   9                              +      !      '      /            )   '   %                                                                                                   -                              !            !      %               '                                                   N�@�N3�O&v%O��BN�*SN�g�O2zQN*��N���Ou;�N���N2�mO_�EO��O!�/OZڙP;�CNj�NghIO&hcO�{9NE��N��O�'Nn�yN�J�O���O��nOX�\N�9WOۡ?O�P��NK��Ng��N�*�O���PߢO��6N��O0��N��N��N��@O��OAͤNچ�NT��N��UN �<O$�XOz��N�DOP��N���  f  :  �    �  �  $    �    �  �  U  �  �  �  �  @  �  �  �  �  �  J    �  
  t  �  �    [  G  O  l  t  7  �  y  �  
  �  '    �  G  �  (  �  �  X  �  
,  �  似j����o�T���49X�#�
�D����o%   :�o:�o<#�
=��#<u<�o<�`B=+<�1<�j=o=�P<�/<�/<�`B=o<��=e`B=C�=L��=��=e`B=��=ix�=,1=,1=L��=u=T��=�%=�C�=�o=�o=�C�=�C�=�C�=��=�\)=�\)=�t�=��P=���=�9X=�v�=�;d=�h�����	 $�����][Zanrqnma]]]]]]]]]]cacgtt���������utogcZ[_gt���������tgc\[Z������
�������
(($"��������������������RQSTabggaTRRRRRRRRRR��������������������������������������������
#+*##
�������!"/;>=;/"NNQX[gt�������vtg[RN	�
#'/.//+#
	TORVW[ehw�������th[T"!)-6BO[]_^[[SOB>6)"��� )BHTWQH>)��$),)����������������������������������������������������������������������������������������������

������������������������a\behjtw}~���|ttkhaa������1>?==5)�vvz��������������{v���
#/6874/)#
����������������������)5BJNUZ[SN)������ 

���������
#/:=DEA</#
�xz��������zzxxxxxxxx;02<HQUYUH?<;;;;;;;;;548<HNU\UTH=<;;;;;;~������������������~)BNew�zgN5)���)BGVUOC74)��������������������������������}x��������������}}}}������������������������������������������������������
�����63;;DHT^aca]VTH;6666��������������������hhmst��������}thhhhh����������������������������������������IGA<#		#/4=BI,*/2<HOTPH</,,,,,,,,hcbfnz����������zqnh�������������������ܼ��ʼּ����߼ؼּʼ�����������������E�E�E�F	FE�E�E�E�E�E�E�E�E�E�E�E�E�E�E��n�zÓÝììëàÜÓÇ�z�u�n�i�a�`�a�d�n���������������������������������������޻F�S�_�l�x�{���x�l�_�S�F�=�D�F�F�F�F�F�F��"�%�.�:�8�.�"��	�����	����������������������������������������������������������������������������������������5�6�A�B�B�A�5�5�(�!��#�(�4�5�5�5�5�5�5������4�0�(������������ۿڿݿ��������������������������������a�m�n�t�q�m�a�^�Z�\�a�a�a�a�a�a�a�a�a�a�B�O�[�h�k�q�p�k�h�[�O�B�6�1�)�(�&�*�6�B�`�m�y���������������y�m�a�`�T�G�Q�T�^�`�A�M�Z�f�s��s�q�f�[�Z�Q�M�A�@�6�:�5�5�A�ʾ׾߾������׾������������������ʿ����ѿ����5�A�5����ݿĿ��������������Z�f�m�j�f�a�Z�V�T�X�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z���������������������������������������Ҿs�������������������s�f�c�]�]�a�f�o�s�ùϹܹ����������ܹϹù����������������#�/�0�;�8�/�#��� �#�#�#�#�#�#�#�#�#�#���¼Ƽʼͼּ׼ּԼʼ���������������������)�5�B�I�N�O�N�B�=�5�)������������������ùìàÕàêìùþ�����������Ż-�:�F�S�U�S�J�F�:�-�(�!���
��!�$�-�-���
�#�0�E�I�M�K�>�#�
���������������������(�5�N�V�b�g�l�a�Z�A�$�������������������������������������{�{�����������������������������������������������������ƳƚƎƉ�|ƁƌƚƧ����(�.�4�A�G�M�Y�X�M�A�4�(����������ʾ׾��������ʾ��������{�z�������O�Q�[�]�[�V�O�B�:�6�B�B�O�O�O�O�O�O�O�Oìù����������ùôíìììììììììì��������
�	�����������������������������������������������ŹŭŧŬŭūŬŰŸ���	��;�H�a�g�`�Z�;�/�"��	�������������	�~�������ɺɺϺѺɺ����������~�l�g�e�p�~�@�G�M�Y�f�i�k�f�b�Y�M�A�@�=�;�<�@�@�@�@�@�M�R�Y�\�\�Z�Y�M�@�4�'�%��"�'�-�4�;�@����������������������������������������¦²¿����������¿²¯¦ £¦¦¦¦¦¦�	��"�#�(�)�%�"��	������������	�	�	�	�������������������������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDxD{D�D��U�b�n�n�{�~��{�x�n�h�b�U�S�I�K�U�U�U�U�����ĽнܽнǽĽ������������������������4�4�@�M�Q�X�M�G�@�4�0�'�&�'�)�2�4�4�4�4�������������������������n�zÇÓàâåäáàÓÇÁ�z�s�n�m�k�h�n���������������*�6�C�O�V�N�<�6�*��EEEEE"E#EEED�D�EEEEEEEEE�ܻ�������� ����ܻлƻû»ǻлӻܻS�_�l�x���������x�l�_�S�F�E�F�G�S�S�S�S 6 G - / J 8 N H E 7 X m  : F 1 R _ g % . C z , h W > `  ?  D H ! A 2 V 9 B 8 = O % 4   < > 4 k < G 4 + m    '  C  u  /  �  �  �  9  �  �  �  �  �  J  �  �  ~  A  �  d    q  �  J  �  �    k  �  �  �  4  �  `  v  �  a  �  �  �  �  %  �     �  �  �  p  �  a  p    �  �  �  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  f  S  >  (  
  �  �  �  {  Y  9  &            %  ;  R  :  %    �  �  �  �  �  |  b  G  +    �  �  �  o  J  +    g  {  �  �  �  �  �  �  �  {  W  %  �  �  _    �  �  �  �  
  
    �  �  �  �  �  �  �  �  �  o  T  3  �  �  �  �  �  {  �  �  �  ~  w  p  g  ]  V  _  d  e  a  V  I  ;  /  #    �  �  �  �  �  �  �  �  �  x  g  U  C  .    �  �  �  �  l    !  "                  �  �  �  �  U  !  �  f   �        (  2  8  5  2  0  -  )  %  !        �  �  �  �  �  �  �  �  {  o  c  W  K  ?  3  &      �  �  �  �  �  �      �  �  �  �  �  �  �  y  m  g  e  b  X  4    �  �  V  �  �  �  �  |  r  i  _  S  D  4  %    �  �  �  �  �  ^  7  o  s  w  z  |    �  �  �  �  �  �  �  |  x  t  o  h  `  W  	�  
�  �  l  �  G  �    �  �  2  T  ?    �    �  P  �  F  �  �  �  �  �  a  >    �  �  �  p  8  �  �    �  >  �    �  �  �  u  i  ]  R  H  ?  6  ,      �  �  �  �  h  d  a  �  �  �     r  �  �  �  u  ^  ;    �  �  q  %  �  `    M  �  �  �  �  �  �  �  �  �  �  �  e  4  �  �  5  �  ;  �   �  @  9  2  ,  %            �  �  �  �  �  �  Z  .     �  �  �  �  {  ~  �  �  u  U  4    �  �  �  {  T  ,     �   �  T  c  r  |  �  �  �  �    r  ^  E  '    �  �  i    �  X  �  �  �  �  �  �  �  �  a     �  �  b    �  J  �  )  A  �  �  �  �  �    w  n  e  T  A  '    �  �  �  �  _  :    �  �  �  �  }  h  T  ;       �  �  �  �  w  Z  5      ;  [  J  @  0      �  �  �  �  t  U  6    �  �  �  �  u  b  [  �  �  
      �  �  �  �  �  �  �  s  V  �  �  0  �  ~  "  �  �  �  �  �  �  �  �  �  �  �  u  k  e  ^  W  U  S  Q  O  	n  	�  	�  	�  
   
  	�  	�  	�  	~  	K  �  �  ;  �  D  �  �  �  �  t  f  O  0    �  �  �  Z     �  �  F  �  �  @  �  �  \  A  3  Y  {  �  �  �  �  �  �  n  J    �  �  C  �  �    �  �  �  �  �  �  �  �  �  �  \  2    �  �  �  �  �  [  -  �  �  �  �  �  �        �  �  �  �  [  6    �  �    {  �  �  [  B  *    �  �  �  �  �  s  Y  >  $    �  z    �  I   �  
`  
�    >  G  C  2    
�  
�  
K  	�  	Z  �    (  8  n  �  �  &  4  @  J  N  O  F  9       �  �  �  =  �  �  6  �  �  :  l  m  o  m  k  i  h  e  b  ]  X  Q  J  B  8  /  (  &  H  j  ~  �  �  /  P  k  s  m  Z  C    �  �  `    d  �  3  �   �  �      $  4  2       �  �  �  �  �  c    �  &  N  u    �  ~  z  y  t  k  V  *  �  �  �  �  �  �  �  s  /  �  +  �  0  \  t  p  W  ,  �  �  g  �  �  L  �  q  �  �  i  �  I  r  �  �  �  �  �  �  �  �  j  ;  �  �  U  �  �  6  �  }    �  
  
  
  	�  	�  	�  	T  	  �  �  :  �  o  �  q  �  0  w  �  r  �  �  �  �  _  U  /    �  �  �  �  i  -  �  �  Y    �  �        %  &        �  �  �  �  �  �  x  i  _  P  >  ,           �  �  �  �  �  �  p  L  !  �  �  �  p  C    �  �  �  �  �  �  t  ]  B  %  	  �  �  �  ~  K    �  �  y  I  �    �  t  �    C  B     �  W  �  �  �  �  h  �     
a  �  �  �  �  �  |  \  7    �  �  �  �  }  [  4      �    E  (      �  �  �  �  �  �  |  h  U  C  3  "       �   �   �  �  ~  r  d  Q  7    �  �  �  n  =    �  �  �  n  G  %  �  �  �  �  �  �  �  �  q  W  C  /      �  �  �  �  �  �  u  X  B  *    �  �  �  ^  !  �  �  R    �  ^  �  `  �  
  ]  �  �  �  k  >      �  �  �  m  <  	  �  �  K  �  =    r  
,  	�  	�  	�  	l  	7  	  �  �  U    �  Z  �  �  A  �  t    b  �  �  �  �  p  >  "  �  �  q      �  -  
q  	�  �  �  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  m  N  /    �  q     �
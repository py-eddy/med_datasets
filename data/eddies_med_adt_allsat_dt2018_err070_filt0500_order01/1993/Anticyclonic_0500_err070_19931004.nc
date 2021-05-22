CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ɺ^5?|�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�y   max       Q      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       >%      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>k��Q�   max       @F��Q�     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @v���R     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @N�           l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @��          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       >��      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B5q      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�z�   max       B5�k      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�Ӑ      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @��   max       C��*      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          I      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�y   max       P�k7      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Q��R   max       ?�tS��Mj      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��/   max       >��      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @F��Q�     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���
=p�   max       @v���R     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @N�           l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @��           �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >n   max         >n      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����A   max       ?�tS��Mj     �  N�      	      	                              A   '         m   $                  M   
      	      %      	   �            Z            %   !   1         T            �         
   O3��O��O[:�N�;�N��N*�!O�CbO�VAO�|N�l�O
�N��O�1�QO���N�¿NA�PxTPz�N�O�N��BN�EN�yPGNN��\O��N��eN�8O��O
UN�M;P<�N�b?O<`�Os�Ps�HO�O)��N���P65O��cO�LANWY�NZa�P�`O*�N$>\N�vO�Nm�N^�=N���N����`B�o��o%   %   ;��
;�`B;�`B<o<o<49X<D��<T��<T��<�t�<�t�<�t�<�t�<���<���<���<��
<�9X<�j<���<���=o=o=�P=�w=#�
=#�
=,1=,1=,1=0 �=@�=@�=@�=P�`=T��=y�#=}�=}�=�%=�o=�O�=�1=ȴ9=���=�;d=�l�=>%LFIN[gru�����tmgb[NLTbegntuuy}~ytige\][T���� ���
#*-++'#
�2256=ABFIOPQOOFB:622ABKO[^[[SODBAAAAAAAA ���
���";HHDH;/"	�������
#/9;:71/(#���������������������uv{���������uuuuuuuu#/<HJSUYUQH<3/+'#&),5985)��������������������z��(OZWB�����~�z����������������������������������������+&,/5<FG</++++++++++���������������������������������`Y]amnrrna``````````v|}���������������zv���������������������� ����������������������������������
5?GOQOB5���)-5575,) #$/<>HTU_USH<9/$#orvzz~����������|zooIGLLN[glonga[NIIIIII�����'(($������@;:8<BIN[]figd[WNFB@rnqz���������zrrrrrrGGN[g����������yg[QG������������������(")2BO[^aba_[WOB61)(��������

	������)/;AWQROB63!�������������������������������

	�����������

������������)5NlwtcN5����������������������������������������������������������������/#

#*////////������)6<84)������������������������������������������������

���������������

�����nnz~����ztonnnnnnnnnxwzz{���������zxxxxvpoqtz�������zvvvvvv����� �������������ÇÓàæìçàÓÇ�z�n�a�`�^�a�f�n�yÆÇ��*�6�B�6�0�*�����������������������:�F�S�_�l�x�������������l�_�S�F�B�0�)�:�����ĽŽнݽ�ݽнȽĽ���������������������������������������������������������úùõù�������������������������"�0�P�a�p�{�n�a�H�;� �
��"�+�1�/���"�m�y���������������y�m�`�T�Q�G�?�D�T�`�m���������������������������������������������������������������������������!����������������������������Z�f�h�f�e�f�m�f�Z�W�S�X�Z�Z�Z�Z�Z�Z�Z�Z�����Ŀٿݿ����Ŀ��������������������"�a�Z�f�b�u�T������������o�s�������	�"�����������!�����������Ƶưưƴ���������������������������������������������	��"�&�"���	���	�	�	�	�	�	�	�	�	�	���Ľݾ�(�A�f�w�}�v�^�M����ݽ����������)�B�E�C�:�������������������������)E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF$F1F=FJFVF\FWFJFCF=F1F$F FF	FFF���������������������������������������������������������������������������������<�H�N�J�H�<�/�-�/�3�<�<�<�<�<�<�<�<�<�<�5�B�O�[�t�[�B�5�����)�0�5�h�uƁƅƉƋƁ�u�u�h�e�\�Y�X�\�]�h�h�h�h���%�(�3�+�(����������������������(�+�5�A�C�H�A�5�(� ���
��������������������������������������������
��!�����������ħĞĚĦĳ�������I�U�b�n�{ŁŇŊŊŇ�{�n�j�b�X�U�K�I�A�I���
��!�����
�����������������������6�O�`�i�q�o�j�[�T�B�6�2����������B�O�S�X�[�^�[�O�B�6�6�2�6�>�B�B�B�B�B�B���ʾ̾׾��ܾ׾ʾ����������������������������ʾ˾׾�����׾ʾ����������������:�x�����������x�:�!�����ֺ���� ��:������'�4�:�@�J�@�4�'������������Y�f�r���������������������t�r�e�Y�Q�Y��#�,�/�4�2�/�+�#���
�
�����������(�N�f�}��~�s�g�f�N�:�&��������l���������������������y�l�[�V�V�`�j�`�l�6�B�L�R�[�]�U�O�B�)��������������)�6àìù������������ùìåàßàààààà�n�h�g�n�p�{ŃŇňŇŀ�{�n�n�n�n�n�n�n�n�@�L�e�~�������������~�r�`�@�3�'�&�)�4�@�/�<�H�U�W�\�Z�U�O�H�<�/�#�"��� �#�+�/�������ľʾ˾ʾ��������������������������{ǈǔǛǡǬǡǠǔǈ�~�{�t�o�{�{�{�{�{�{D{D�D�D�D�D�D�D�D�D�D�D�D�D�DvDiDeDcDjD{�����������������|�����������������������f�r�������������������r�p�f�`�f�f�f�f�ʼּ����������ݼּʼ��ʼʼʼʼʼ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� X c F W I _ ^ ' \ C H v + / G z ? X J 4 ^ E Y H "  K n . @ 4 U 7 : ) ^ A T P 6 n . C o W D 3 X =  D t ` K  �  �  �  �  9  j  �  !  a  �  h  O  �  �  w  �  O  b  �    o  �  5    f  �  S    �    >  �  �  �  �  B  C  V  �  �  5  \  �  �  �  �  v  [  �  �  <  �  �  ;�t�%   <T��<t�;�o<�o<�1=C�<49X<D��=�w<u<��=��-=e`B<ě�<��
>J=aG�<���=t�<�`B<���<�/=��=\)=D��='�=8Q�=���=Y�=D��>M��=]/=�7L=]/>C�=�C�=��=�7L=�9X=��=�G�=��=���>�+=�Q�=�9X=�S�>��==�>J>$�B	D/B	HuB$��B0�B�XB�A���BU�B�B
nB��B��BD�BϹB�B��B��B"[�B�Bx5B�
B{�BgB�*B�kBcBƫB��B�B�BHB£B
hB�B}�B6=B��B!�B#E�B�cB�!B+��B��By�By�B�?B��B5qB��BĵB. B�(B4<B��B	�B	�NB$JTB?�B��B=�A�z�B��B��B
}�B�EB�GBZBz!B��B�`B��B"OB>_BE3B?�B��Bg:B��Bp&BB�B�.B��B�MB�VBm�B��B	�mB�*B��B/�B@)B!��B#H�BG�B�2B,.�B�@BD\B@B��B��B5�kB�B��B@B��B?�B��A�N�A� y@�NVA%ٝA/;iA���A�ݩAk��A�^A��A���A?�=Av:EA��B.}A�^
A�I�A3r5A�-C�xVC�ӐA�o\A�j�Aç�A��]B��A��/A�NB	tA�A�<8A��BA׹�A���AN�RAO��@��@űY@�h:A��A���A�PA�`�A͏�A���?���AÔeAM��B�~C��@��U@嚷AaC��A�DDA�py@��A%�A.��A΃A�~[Al�kA�|�A���A҈�A? �Au�A���B<�A���A��A2�A�uSC�yC��*A�8�A�O�A�ckA��#B��A�|�A�KyB��A�|=A�TA�|�Aׅ�Aئ$ANf�AQ�@|�x@���@��!A�+�A�RA�UA�u�A�|A�rc@��AË�AN�oB�0C�ט@��j@��A�	C��      	      
                              A   '         n   %         	         M   
      
   	   &      	   �            Z            %   "   1         T            �   	                              -                  !   Q            ;   '                  )               !         '            7            3      !         -                                                               !   I            #   '                                                      )            '               #                        O!�O��O[:�N�q"N��N*�!OgQ=Ob��O�|N�l�NV�`N��O�1�P�k7Oa6N�¿NA�O�O�>�N�O�N��BN�EN�yO�1�N���O��N��eN�8O���N�;�N�M;O�0N]�O0u�Os�P�O�O�kN���O��oO^��O.�N4�NZa�O��4N�kN$>\N�vOx�tNm�N^�=N���N��  �  �    �  w  �  �  �  �  k  �  �  K  o  )  \  �  	s  5  _  V  a  �  �  	�  �  �  �  ~  �  Q  a  �  �  �  �  �  �  {  �  �  �  k  �  �  	�  �  0  O  i  c  �  �  D��/�o��o:�o%   ;��
<#�
<49X<o<o<�j<D��<T��<�1<���<�t�<�t�=��<�1<���<���<��
<�9X<�j=L��<�/=o=o=�P=49X='�=#�
=�
==49X=0 �=0 �=�O�=@�=H�9=P�`=}�=�C�=��
=�%=�%=���=��=�1=ȴ9>��=�;d=�l�=>%JKN[gpt����tngb[PNJTbegntuuy}~ytige\][T���� ���
#*-++'#
�3267>BBCGOOPOMCBA633ABKO[^[[SODBAAAAAAAA ���
#/;BDB>A;3/)"����
!#/5974/#���������������������uv{���������uuuuuuuu../<HJOHD<6/........&),5985)���������������������������OVSB)����������������������������������������������+&,/5<FG</++++++++++����������� ���������������
 ��������`Y]amnrrna``````````v|}���������������zv���������������������� ������������������������������	)5:@EFEB:5)	)*3565*) #$/<>HTU_USH<9/$#orvzz~����������|zooIGLLN[glonga[NIIIIII������#$ �����<;9>BKN[\eghgc[UNB<<rnqz���������zrrrrrrYWX]gt����������tg]Y������������������)4BO[]`a`^[UOEB:62*)��������

	��������)0:HKCFB6)������������������������������

������������

�������������5BN[``Y5���������������������������������������������������������������/#

#*////////������)00/)"������������������������������������������������

���������������

�����nnz~����ztonnnnnnnnnxwzz{���������zxxxxvpoqtz�������zvvvvvv����� �������������ÓàãêæàÓÇÄ�z�n�a�^�a�h�n�{ÇÌÓ��*�6�B�6�0�*�����������������������:�F�S�_�l�x�������������l�_�S�F�B�0�)�:�����ýĽнݽ߽ݽнŽĽ���������������������������������������������������������úùõù�������������������������H�T�a�m�s�u�m�a�T�H�;�,�"��"�(�/�6�;�H�m�y�������������������y�m�`�V�E�H�T�`�m����������������������������������������������������������������������������	��������������������������������Z�f�h�f�e�f�m�f�Z�W�S�X�Z�Z�Z�Z�Z�Z�Z�Z�����Ŀٿݿ����Ŀ����������������������	�"�;�O�V�S�Z�Q������������v������������������������������������ƸƸƻ�������������������������������������������	��"�&�"���	���	�	�	�	�	�	�	�	�	�	���(�C�M�S�Q�I�7�(������׽Ͻѽݽ���)�5�B�A�8�������������������������)E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF$F1F=FJFVF\FWFJFCF=F1F$F FF	FFF���������������������������������������������������������������������������������<�H�N�J�H�<�/�-�/�3�<�<�<�<�<�<�<�<�<�<�B�N�g�y�{�t�g�[�N�B�5�)�$�#�%�-�5�B�h�uƁƁƆƅƁ�u�m�h�g�\�[�Y�\�`�h�h�h�h���%�(�3�+�(����������������������(�+�5�A�C�H�A�5�(� ���
����������������������������������������������
����
��������ĳĭĢĥĳĿ�����U�b�n�{�ŇŉŉŇ�{�p�n�m�b�Z�U�M�I�U�U���
��!�����
����������������������)�6�B�O�[�`�a�]�[�O�B�6�)�!������)�B�O�R�W�Y�O�B�:�6�4�6�?�B�B�B�B�B�B�B�B�ʾ׾߾�ھ׾ʾ������������������������ʾ������ʾ˾׾�����׾ʾ����������������!�:�_�x���������l�F�:�!������� ���!������'�4�:�@�J�@�4�'������������r�����������������������v�r�i�f�]�f�r��#�,�/�4�2�/�+�#���
�
���������(�5�A�N�Z�c�s�t�q�l�g�Z�N�-��������������������������y�l�i�`�^�^�`�l�y���)�6�B�K�K�B�=�6�)���
�� �����#�)ìù��������ùìæáìììììììììì�n�h�g�n�p�{ŃŇňŇŀ�{�n�n�n�n�n�n�n�n�L�r�~�����������������~�r�e�L�7�1�4�@�L�<�H�K�U�Y�W�U�K�H�<�/�#�#�"�#�&�/�7�<�<�������ľʾ˾ʾ��������������������������{ǈǔǛǡǬǡǠǔǈ�~�{�t�o�{�{�{�{�{�{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DuDpDqD{D������������������|�����������������������f�r�������������������r�p�f�`�f�f�f�f�ʼּ����������ݼּʼ��ʼʼʼʼʼ�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E� Z c F R I _ 2 # \ C & v + , @ z ? = H 4 ^ E Y H  ! K n . ? 0 U " @ $ ^ 2 T M 6 f / % g W F ' X =  D t ` K  |  �  �  �  9  j  �  �  a  �  \  O  �  H  �  �  O  �  n    o  �  5    B  �  S    �  �    �  '  ~  s  B  �  V  <  �  �  �  v  N  �      [  �  �  <  �  �  ;  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  >n  �  �  �  �  �  w  c  O  @  f  f  V  7  3  �  �  \    �  K  �  �  �  �  �  r  u  w  �  �  �  �  �  }  z  g  O  6             �  �  �  �  �  �  �  �  �  ~  �  �  �  x    �  Z  �  �  �  �  �  �  �  �  �  �  �  x  b  J  -    �  �  �  t  w  v  u  t  s  q  m  i  f  b  X  H  8  (      �  �  �  �  �  �  �  �  �  h  N  6      �  �  �  �  x  [  1  �  �    �  s  �  �  �  �  �  �  �  �  �  �  �  a  )  �  �  J   �   o  S  l  |  �    t  [  =    �  �  �  k  .  �  �  6  �  p  7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  f  a  [  V  P  I  B  ;  4  .  *  &  "        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  F    �  ^  �  �  "  �  �  �  �  �  �  �  �  �  }  j  X  E  4  (         �   �  K  H  C  =  0  !       �  �  �  �  �  �  z  Q  "  �  �  �    :  l  j  M  %  �  �    F    �  �  b       �  �  j  u  �  �    '  '       �  �  �  �  Z  !  �  t  �  T  �  �  �  \  Q  E  9  *      �  �  �  �  �  �  {  t  l  g  n  u  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  u  n  g  _  �  �  H  �  �  	)  	K  	c  	s  	m  	L  	  �  w    �  �  �  �  �  (  4  2  (        �  �  �  �  a  >  &  �  �  U  �  R  �  _  U  J  ?  4  *      	  �  �  �  �  �  �  �  q  ;    �  V  9       �  �  �  �  �  l  9  �  �  o    �  X  �  o   �  a  R  D  5  '        �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    k  W  C  /  $  �  �  	@  	m  	�  	�  	�  	s  	C  	  �  w  #  �  7  �  �    G  �  �  �  �  �  �  �  �  �  �  k  U  ;    �  �  r  <     �  �  �  �  �  �  z  b  I  +    �  �  �  h  '  �  �  X  I  �  �  �  �  �  �  �  �  |  g  K  *  	  �  �  k  $  �  �  0   �  ~  {  x  o  c  U  B  /    	  �  �  �  �  �  �  V  .    �  �  �  �  �  �  �  �  �  �  k  J  "  �  �  ~  +  �  ;  y   t  K  O  I  :  *      �  �  �  �  �  �  e  E  &    �  �  )  a  \  X  Q  G  9      �  �  �  �  v  J    �  �  �  v  R  �  �  .  �  �  4  `  }    \    �  �      �  v  
�    �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  O    �  �    �  �  �  �  �  w  f  S  ;    �  �  �  Q    �  �  >  �  �  5  �  �  �  �  y  q  d  V  D  0      �  �  o  0  �  �  }  D  +  �  �  �  �  �  �  �  g  2    �  �  (  �     ;  �  V  �  �  {  k  c  a  Y  M  <  (    �  �  �  �  Y    �  �    Z  ]  `  x  r  i  \  L  7      �  �  �  �  �  `  5    �  �  �  �  �  p  R  0    �  �  �  �  s  Y  A  )        #    c  f  _  o  �  �  �  ~  a  ;    �  �  j    �  n  �  M  �  �  �  �  �  �  �  �  v  O  '  	  �  �  �  �  u  2  �  s  X  A  =  V  X  V  Y  `  j  c  C    �  o    �      �  =  \  �  �  �    i  R  =  '    �  �  �  �  P    �  �  �  L    �  �  b  E  5  "  
  �  �  �  �  {  ^  @  #    �  �  @  e  	!  	t  	�  	�  	�  	�  	�  	�  	L  �  �  {  g  ,  �  N  �  T  �  {  �  �  �  �  �  �  �  �  x  Z  6    �  �  F  �  d  �     �  0  )  "        	    �  �  �  �  �  �  �  �  �  �  �  �  O  6      �  �  �  �  g  A    �  �  �  E  �  �  n  D  @    \  �    X  i  Y  5  �  �  $  b  r  d  !  U  5  �  �  �  c  E  &    �  �  �  g  8    �  �  k  4  �  �  v  /   �   �  �  �  u  \  .     �  �  �  �  �  �  �  �  �  �  y  m  a  V  �  �  �  �  �  g  I  ,    �  �  �  �  �  k  ?    �  �  �  D  4  #      �  �  �  �  �  �  �  p  ^  K  9  )      �
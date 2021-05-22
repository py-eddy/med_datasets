CDF       
      obs    6   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ǍO�;dZ      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =�1      �  \   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>޸Q�   max       @F������     p   4   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @v�=p��
     p  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @M            l  1   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @��@          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >S��      �  2X   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B4�S      �  30   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�a^   max       B4�f      �  4   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?09�   max       C���      �  4�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?>��   max       C��y      �  5�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  6�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  7h   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  8@   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P�+|      �  9   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?�Ov_�      �  9�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       =�      �  :�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @F������     p  ;�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�     max       @v�=p��
     p  D   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @M            l  L�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�-        max       @�C�          �  L�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  M�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?�Ov_�     �  N�            2                  X      ,                  
   (      _   �   J         A         @   !   /         :   @   /      
   J                   $      %   "         O   %      O�N!N!ɰPC�O�mOn��NC�)N���O��P��N�аO��N��O�� N��NH��OE	�NC�sO��N+t�P-܏P���O���N0��NY�GP��N1��N^�P6ИP�O���N���O'�O�\IO�6WP�$�O]ͺN�%0P
�OQ�9N�HO�2vN�_NI<O�v�O��O���O��O��O2�O��mO�0�N���Ol,��`B��9X��o��o:�o;�o;��
<o<#�
<e`B<u<�t�<�t�<���<��
<��
<�9X<�j<�j<�j<ě�<ě�<ě�<ě�<�/<�h<�h=C�=t�=�P=��=��=�w=#�
=#�
='�=0 �=49X=49X=8Q�=8Q�=D��=D��=L��=T��=Y�=Y�=e`B=e`B=e`B=q��=q��=u=�10..036;@BOPRTUROKB60��������������������)-//)OPWanz�����������pXO�����!)+#
�������QOV[egpt��������tg[QZW[[bgtuttg[ZZZZZZZZ)6<<=6)������������������������BLYjqyyoJ���������������������"%,/<HUanvvpiaUH</'"�������������g`\_\ht�����������tg������������������������������������`\\aimrz�������zmf`ntv������tnnnnnnnnnn����������������

#,)#










���)58/-43*)����)5Bg�������[B,"���������������������������������������������������������������
#/6?LM</#
 �#%$#��)))#������knv��������������znk!#-/5BJ[ip����t[N5)!#/6BOU[^^]XOB6.������

���|u�����������������|�������/6:/# ����������
#/<HRUTQH<#
���������!������'#"'/;HTY[ZWTPH;70/'"/2;@<;6/)"��������������������
"#)/6773/#
	 
####"
				2AY[ghlmh^N5)ba_egt���������ztgbb������������������������%'%!�������� )6DGDB6)�%)6BEJID@@0)���������������������������������������#$#
������������#-6?B?)����EFOht��������tni`TOE������� �����������/<>HJGGG</#�l�y�������������������������y�q�l�`�_�l�'�3�<�9�3�'�����'�'�'�'�'�'�'�'�'�'�G�T�`�a�b�`�T�G�A�B�G�G�G�G�G�G�G�G�G�G���5�I�Y�c�g�q�s�Z�N�A�����������(�=�G�B�>�5�(����������������(����������������������������������������àìùùùøïìàÞÔÞààààààààÓàçáãèàÓÇÀÂ�ÇÇÓÓÓÓÓÓ�f�s�~���������s�f�Z�Q�T�X�Z�]�f�f�f�fƚƳ������G�J�H�=�$�������Ƴ�u�?�J�^ƚ��������������������������������������������������	���������������������	��"�.�;�G�T�T�T�N�G�;�.�"���	��	�	���4�A�M�W�R�N�T�S�M�A�=�(��� �������%�#���������������������������������������������������������N�Z�g�y�����������s�g�Z�N�A�:�5�+�6�A�NFJFTFVF^F^FVFJFIF>F@FJFJFJFJFJFJFJFJFJFJ�������
����ܻ������������ûܻ��������������ܼܼ߼������������0�A�O�V�c�a�U�<�#����ĳĦĲľ��������6�O�hčĦĨĦėĄ�t�[�6�"� �����)�6�Ϲ��'�5�;�:�2�'����ܹù������������ϽĽнݽ�����ݽڽнĽ��ĽĽĽĽĽĽĽ�E�E�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��;�H�T�a�m�z�}�|�o�a�;��	������������;�����������������������������������������Z�f�h�h�f�b�Z�Z�Z�M�I�M�O�T�Z�Z�Z�Z�Z�ZŔŭ������������������ŹŭŎņ�~��ŇŔ����;�T�_�g�a�\�T�H�=�"�	��������������������ʾ׾ܾ�ݾ׾˾��������~�~���������A�M�Z�f�s�t�������{�s�f�Z�M�C�A�5�@�A�������������������������~�}�r�n�r�~�����)�5�B�[�i�j�f�[�\�[�N�L�4�)����
��)D�D�D�EEEE"E*E/E0E*EED�D�D�D�D�D�D���������/�=�@�=�/�"�	�������������������#�0�<�I�T�U�T�I�<�0�#���
����
���#�T�a�k�m�t�y�z�}�z�m�m�a�Y�T�M�K�T�T�T�T�a�nÇàù����������ìÓ�z�`�Y�W�S�V�]�a���	��"�+�.�;�;�.�(��	�����ݾ׾׾�`�m�y���������y�o�m�`�W�T�P�T�_�`�`�`�`��������(�A�E�N�A�(����ؿɿɿѿ���g�s���������������������t�s�m�g�c�^�g�g�ʾ׾ܾ����ݾ׾Ծʾʾʾʾʾʾʾʾʾ����
����
������¿²§²µ¿��������������!�.�6�-�������޺ۺ���ٺں�r�~�������ºɺٺкϺɺ������e�Y�L�A�N�r���������������������������|�{����������M�Z�f�p�p�f�a�Z�M�A�(� ����(�-�4�A�M�Z�[�f�l�f�d�S�M�A�4�'��!�*�4�>�A�M�X�Z�f�r�������������r�^�M�@�4�+�,�3�@�M�Y�f�4�@�B�K�F�;�4�������߻������"�4�"�.�8�;�G�G�L�G�;�.�"������"�"�"�"��*�6�C�M�C�2�,�*������������������ X , , ) : A 3 U + H 2 - _ F X W W [ O I P 2 I t 0 D * e @ - # k F ? P # / [ " @ G ^ U ` A E T \ " U 9 Q 9 \  \  5  ?  v    �  T  �  (  �  �  0    �  b  I  �  �  �  \  �  `  Z  p  o  �  G  �  H  �  6    b  �  �    �  �  �  �  �  }  >  �  �  $  �  s  �  �  O  d  �  (�T����o�o=8Q�<�j<��<�o<u<���=��`<���=}�<���='�<�j<�j=,1=o=}�<�/=��>S��=ȴ9<�`B=�w=��=+=#�
=���=�\)=�{=D��=m�h=ȴ9=��=�9X=�o=]/=��=�7L=aG�=���=T��=e`B=�-=���=�E�=�Q�=�t�=��P>I�=ě�=�t�=��`B(RB!�B�BӡB�]B	�7B	7�B�B$�B��BK�B�BG{B�B�B��A�"�B��B#B$��B�0B��B�B"̿B�LBy]BQ�B�RB�{B`�B4�B$	�B�kB�sB��BjTA��!A��BgB~�B-B�0B
�B4�SBC�B�aBs[B��B�TB�;B5B+wB-d�B B@�B!=�B�]B��B��B	�_B	?�B�OB!BB�<BClB:IB>^B�B)�B�A�}�B�B#@�B%,B�WB��BAxB"��B��B?�BF�B��B>�BBB@B#˖B@B� B�B��A�v�A�a^B��B�pB��B5�B	�jB4�fB�BrB �B��B�'B�QB@�B��B-@6B?�A�?�=�Af�A��3A�͸A��A�:XAʪ.AB|�B'Aru@A�A`ݨA8��A��Ar�A���C���@���A��A�CA��?09�A)�MC�~)A��-A���A?$kA�:A�?�AL��A?�G@W�A���C�W�A���A�R�A���Aʖ9AZX�Ak!,A�|�A�A�AR��A��M@Z�%@g�A�f�A;ڷA;��@ި@�h�AaOA��WA�?�HAf03A�z4A�x�A��XAˇTA�rAC�B�iAq��AҀfA`��A7
�AՃ+Ar�A���C��y@�EBA�*A�`!A٧�?>��A(��C�xA�|PA�q<A>�SA�>�A�i1AL��A>'.@ YA���C�XA�>�A�BA�q�A� NAZ��Al��A�y�A��HAR��A�aZ@\+�@FA��A:��A;2@�wN@���A`�-A�~�            2                  Y      -   	               
   (      _   �   K         B         A   "   0         ;   @   0         K                   $      %   #         O   &                  %   #               E                           !      /   5   %         +         +   %            #      5         '         '               #            %                        !               9                                       !         %         '   #                  5         !         !               !            #         N迳N!N!ɰO�r�O�v?O@0�NC�)N���NarfP�+|N�аN��N��O��UN��NH��O�9NC�sO(�N+t�O�o�O�`�O�@*N0��NY�GO��N1��N^�P�O��QO?`�N���O'�Ou�-O�6WP�$�O]ͺNuU�O�.~O+�+N�HO���N�_NI<ORX�On�QO�_@O�Of�O!�}O��Om��N���Ol,  |  �  `  �    �  =  �  h  �  +    �  �    �  m  E  -  /  �  �  	4  �  �  �  �  �  	=  
  �  Y  \  B  i  p  ]  i  	v  �  �  %  V  �  <    �  �    �  �  m  9  �������9X��o<t�;D��<o;��
<o<�t�<���<u=#�
<�t�<�1<��
<��
<���<�j=\)<�j=m�h=�=�P<ě�<�/=�P<�h=C�=49X=0 �=H�9=��=�w=]/=#�
='�=0 �=<j=ix�=D��=8Q�=Y�=D��=L��=}�=aG�=m�h=�+=ix�=ix�=�t�=�o=y�#=�12/0156?BJOPRSQOHB622��������������������)-//)a_cgpz�����������zpa�������
'#
������TY[`gt���������ztg[TZW[[bgtuttg[ZZZZZZZZ)6<<=6)����������������������	5BbopfSC5����������������������-./5<HPU[XUH</------�������������b_a`ht�����������tjb������������������������������������__chmz{�����zsmhba_ntv������tnnnnnnnnnn����������������

#,)#










���&))(&�����4213:BN[gt{�~ym[NB:4����������������������������������������������������������������
#/49FD</#
��#%$#��)))#������st{���������������zs'&)25BN[gw����f[N5)''$&-6<BOUYZZWQOB6-)'������

���|u�����������������|�������� ����������
#/<HRUTQH<#
���������!������'#"'/;HTY[ZWTPH;70/'"/94/'"��������������������	
!#$/4552/(#
		 
####"
				#-9HN[`giia[NB5) ba_egt���������ztgbb������������������������������� )4CEB6/)�)6BDFEAA><+)�������
�������������������������������
"$#!
��������")477+�����YPOWht���������tkhdY������ �������������/<>HJGGG</#�l�y�����������������������y�t�l�d�c�l�l�'�3�<�9�3�'�����'�'�'�'�'�'�'�'�'�'�G�T�`�a�b�`�T�G�A�B�G�G�G�G�G�G�G�G�G�G���5�A�N�V�Z�]�]�Z�N�A�(�������������(�5�A�<�;�3�(������������������������������������������������������àìùùùøïìàÞÔÞààààààààÓàçáãèàÓÇÀÂ�ÇÇÓÓÓÓÓÓ�f�s�������w�s�f�b�`�b�f�f�f�f�f�f�f�fƧ������"�:�<�7�����ƳƎ�u�f�T�\�pƊƧ��������������������������������������������������� ������������������������	��"�.�;�G�T�T�T�N�G�;�.�"���	��	�	��4�A�M�T�O�M�L�P�N�A�(���
��������%�#���������������������������������������������������������Z�g�s���������s�g�Z�N�A�5�/�5�9�A�N�U�ZFJFTFVF^F^FVFJFIF>F@FJFJFJFJFJFJFJFJFJFJ������������ܻлû������������ûлܻ����������ܼܼ߼������������#�1�:�;�5�0����������������������
��B�O�[�h�t�{���y�p�h�[�O�B�6�-�+�.�6�B����'�-�3�1�&�������Ϲù������۹�Ľнݽ�����ݽڽнĽ��ĽĽĽĽĽĽĽ�E�E�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��H�T�a�m�w�y�w�i�a�H��	���������
��;�H�����������������������������������������Z�f�h�h�f�b�Z�Z�Z�M�I�M�O�T�Z�Z�Z�Z�Z�ZŔŭ������������������ŹŭœŋŃłņŇŔ����#�;�N�X�X�Q�E�;�/�"���������������������ʾԾ׾پԾʾ����������������������A�M�Z�f�s�t�������{�s�f�Z�M�C�A�5�@�A�������������������������~�}�r�n�r�~�����)�5�B�N�[�_�c�_�[�N�B�5�)�������)D�D�D�EEEE"E*E/E0E*EED�D�D�D�D�D�D���������/�=�@�=�/�"�	�������������������#�0�<�I�T�U�T�I�<�0�#���
����
���#�T�a�f�m�r�w�m�a�[�T�P�Q�T�T�T�T�T�T�T�TÓàù��������ýìÓÇ�z�j�a�]�[�a�n�zÓ���	��"�'�.�1�.�"��	�������۾��`�m�y���������y�o�m�`�W�T�P�T�_�`�`�`�`�������(�/�9�5�)������ۿԿϿӿݿ��g�s���������������������t�s�m�g�c�^�g�g�ʾ׾ܾ����ݾ׾Ծʾʾʾʾʾʾʾʾʾ���������
����
�������������������������!�+�-�4�+�������޺�������~���������źɺƺ��������~�e�Y�L�H�V�r�~�����������������������������������������4�A�M�Z�f�m�n�f�_�Z�M�A�4�(����(�0�4�M�Z�f�j�f�b�[�R�M�A�4�,�(� �#�(�,�4�A�M�r�������������r�Y�M�@�4�0�1�8�@�M�Y�f�r��'�4�@�F�C�8�4�'��������������.�6�;�E�G�J�G�;�.�"�����"�"�.�.�.�.��*�6�C�M�C�2�,�*������������������ I , ,   5 > 3 U < > 2 - _ B X W S [ L I >  D t 0 C * e ? .  k F  P # / Q   : G M U ` / L R F $ T 6 T : \    5  ?    �  �  T  �  r  �  �  �    f  b  I  t  �  z  \  ~  @  �  p  o  _  G  �  �  �  �    b  �  �    �  �     �  �  x  >  �  �  �  n  `  �  �  �  �  �  (  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  s  w  z  {  z  z  v  p  g  Z  L  ;  (    �  �  �  �  �  s  �  �  �  �  �  �  �  �  �  �  �  �  �    E  v  �  �  #  _  `  _  ]  \  [  Y  X  W  V  T  R  O  L  I  F  C  @  =  :  7  �  .  ]  t  �  �  |  Y  /  �  �  �  ]  )  �  �  �  6  5   �  �        	  �  �  �  �  �  �  �  �  l  N  0    �  �    �  �  �  �  �  �  �  �  �  �  j  D    �  �  w  $  �  k  X  =  %    �  �  �  �  �  �  �  �  �  �  q  ]  H  1      �  �  �  �  �  �  �  �  �  �  �  �  �  Y  *  �  �  �  �    j      )  3  <  C  K  V  `  f  g  d  W  7  �  �  0  �  p  b  e  �  �  �  �  w  `  U  X  U  F  &  �  �  �  G  �  '  7  L  +  !        �  �  �  �  �  �  �  �  l  W  E  4  #      �  O  �  �  8  y  �  �  �      
  �  y  �  q  �  &  a  �  �  �  �  |  u  m  b  X  J  ;  )    �  �  �  |  D     �   �  j  ~    |  t  u  u  n  f  ]  Q  B  (  �  �  �  N    �  �            !  $  '  *  -  0  2  5  7  5  4  2  0  /  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  h  Y  J  -  O  h  l  g  W  @  &    �  �  �  �  n  G    �  �  w    E  2      �  �  �  �  �  �  v  m  }  g    �    �  	  �  �  �  �  	    '  ,  )      �  �  o  S    �  L  �  G  �  /  (         
    �  �  �  �  �  �  �  �  �  �  �  �  �  	`  
.  
�  F  �  �  �  �  �  �  �  N  
�  
r  	�  �  �  k  	  �  
�  �  .  �  g    �  +  �  �  �  �  �  7  ^  B  �  
D  �  ^  $  �  	  	1  	1  	   	  	  	  	  �  �  �  ;  �  M  �  r    S  �  |  x  s  o  g  U  D  3  "    �  �  �  �  �  �  y  b  K  �  �  ~  T  .  
  �  �  �  �  �  �  t  Y  8    �  �  �  k  }  �  �  �  �  �  n  F    �  ~  7  �  `  �  c  �  ,  Y  �  �    }  |  {  w  l  `  U  I  ;  *      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  S  8    �  �  �  u  D    	  	,  	;  	9  	#  �  �  �  -  �  s  2  �  �  -  �  %  c  y  �  �  �    	  	  �  �  �  �  a  8    �  �  r  4  �  �  +  &  y  �  �  �  �  �  �  �  �  }  ]  7    �  |  �  2  A     �  Y  V  R  N  I  <  0  $    
  �  �  �  �  �  �  ~  e  *  �  \  ?  $    �  �  �  z  P  %  �  �  �  �  |  5  �    p   �  �  �    /  >  B  <  0      �  �  b    �  �  ^  �  �  &  i  f  Y  E  "  
�  
�  
w  
/  	�  	�  	=  �  h  �  M  �  �  �  g  p  U  2    �  �  �  Z  &  �  �  o    �  9  �  �  <  �   �  ]  +  /  +    "  "      �  �  �  y  G    �  �  X    �  ^  S  J  [  f  U  H  A  ;  4  (      �  �  �  �  {  U  0  	  	K  	e  	s  	u  	q  	b  	D  	   �  �  {    �  ,  �  �    �  B  �  �  �  �  �  �  �  �  �  c  3  �  �  q  .    �  j    �  �  �  �  �  {  Z  8    �  �  �  �  b  >    �  �  �  \  �  �      %        �  �  �  �  �  m  9  �  �  H  �  5  z  V  H  ;  .  !      �  �  �  �  �  �  n  U  =  $     �   �  �  �  �  �  �  �  �  �  }  l  T  4    �  �  �  y  N  #   �  �      +  7  ;  ;  7  +      �  �  u  4  �  �    m  �  �  �    �  �  �  ~  F    �  �  �  �  t  <    �  �  C  �  �  y  �  �  �  m  >    �  �  �  ^    �  j    �    \  ]  y  �  �  �  �  �  �  �  �  �  s  9  �  �  F  �  "  _  �  �        	  �  �  �  �  �  �  �  �  �  �  �  �  �  q  U  (  �  �  �  �  �  �  �  �  j  H  "  �  �  �  s  R    �  �   c    �  �  �  �  �  t  :  
�  
�  
  	|  �  '  _  �  �  l  �  G  [  j  k  j  \  H  *  �  �  �  J  �  �     �  �  �  �  #  �  5  7  6  /  $      �  �  �  �  �  �  �  f  <    �  �  v  �  s  X  B  .      �  �  �  �  �  f  3  �  �  �  r  -  4
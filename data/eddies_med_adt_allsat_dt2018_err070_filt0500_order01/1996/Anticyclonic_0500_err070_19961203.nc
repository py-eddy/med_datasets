CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?���E��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nk�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\)   max       >         �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E�ffffg     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vr=p��
     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��h   max       >�J      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B*�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�[   max       B*�8      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Ż7   max       C�p�      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�h�   max       C�nV      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          5      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          5      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��m   max       P��      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Z��   max       ?ᝲ-V      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\)   max       >333      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @E�          �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=p�    max       @vr=p��
     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @S            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�k        max       @���          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B�   max         B�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�>�6z�   max       ?��ߤ@     �  QX            	      $      $      	      
                  "         ?               	   >   �         .            
      .         6               
   
   0         E   �   �            
   4Nh��N���NaN�^O���Px�nN�=�P8��N��Ny+�N��N]@�N�blP8`=O���N��Nm��Ok�N��N�d�P��OWoNtr�N�jIN��LN��P��PT��N2��Ou��P�O���O��O���N��vOq�P�N�O3O���O��Ok`�OvcOi�gN1�zO�WN�-DP'k�O'�*N���O��oO���P#��N�&[N��rNJިNk�OT(�\)�o��o��o%   ;o<#�
<49X<D��<T��<T��<e`B<u<�o<�o<�o<�C�<��
<�9X<�9X<�j<�j<���<���<���<�/<�/<�`B<�h<�h<�<�<��<��<��=C�=\)=\)=<j=D��=D��=L��=P�`=T��=T��=q��=y�#=�%=�%=���=�{=�^5=�^5=�
==�`B=>   WVQ[dgotwtjgf[WWWWWW646BOVZWOOBA<6666666��������������������@BGMNP[\__][NKGB@@@@flt���������������f(--J[h{��������t[>0(eeegqtv������tlgeeee����������������������������������������DOR[hrtkh[ZODDDDDDDDvuu�����������vvvvvv��
"!
���������DGLNU[giqmg][NDDDDDD�����
#/8HNI</#����������������������������������������������������������������

������������������������/./4<HLU`abaWUH<;///���� *:HTWUH</������������������������ffhkt������thffffff�������������������������

	�������������������������������)K[_[]XN5���������05=OMDB>)����������������������������
#,*!
�������GMKQQWg���������t[UG4206=BHN[^chmqrhOB64\abca`WUH?<689<?GHU\���)5@D@72$)/)������������������������������������������'*5BN[�����tg[N5)�����������������������	"/;HT`VTKH;/	�)16=BFOY^VB6)��������������������"/;CFB>;0/" ����)59?;5)������������������������������
�����������������������������������).+�����poqrt������������ytp�������������������������������������������������
������z����������������}yz	 
"#*+)##
		tpoosz���������ztttt`ZUWajnnnnja````````��������������������������
"##
������������������������������������������������
�������ݽܽ۽ݽ����������������������������������ĿĿĿ������������������*�,�*�!������������������@�Y�b�l�r�w�}�r�e�Y�@�3�'�������'�@�������׾ݾ׾����������f�F�D�I�E�I�Z�s���a�n�z�ÇÉÇÂ�z�n�a�U�M�J�U�[�a�a�a�a�s�������������������Z�5�����/�F�Z�sE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��H�P�U�[�\�U�H�<�;�;�<�?�H�H�H�H�H�H�H�H�/�<�H�K�O�H�E�<�/�'�#�"�#�(�/�/�/�/�/�/�����������������������������������������U�b�n�r�{��|�{�n�b�\�U�T�R�U�U�U�U�U�U�/�H�T�a�o�m�e�^�T�/���������� ���"�/�����'�7�@�G�K�@�4������ܻڻܻ����û̻л׻лû������������ûûûûûûû�����������������ùóïöùÿ�������������H�U�a�n�z�}�v�w�v�n�a�U�H�<�+�#�"�+�<�H�)�6�B�O�O�O�J�B�6�.�)������)�)�)�)�����������������������������������뿟�����ʿ������y�\�G�;�.�%�"�%�6�`�{�����������������������������������ÿ�����һ-�:�=�F�Q�L�F�<�:�7�/�-�)�,�-�-�-�-�-�-���!�-�2�-�)�-�:�<�A�;�:�9�/�!�����3�@�L�Y�b�e�i�e�Y�L�@�9�3�,�3�3�3�3�3�3��*�2�6�C�G�O�O�C�?�6�*�!�������ƁƧ���������������ƚ�u�c�S�U�T�\�uƁ�#�<�I�U�Y�X�I�D�0����ĳĪĳĺ�������
�#�)�5�B�E�I�B�5�)�!�&�)�)�)�)�)�)�)�)�)�)����&�������������������������������	��"�/�9�@�?�6��	����������������л�������'�(��������ܻǻ»Ż̻��)�������������������"�)�5�7�6�)�ݿ������	������ݿѿĿ��������ѿݾA�M�Z�f�j�j�f�Z�Z�Z�M�A�A�<�A�A�A�A�A�Aìù����������������ùìàÓÌÈÓàêì��O�tďěĖČ�]�[�O�B�"���)�.�.����zÇÈÇÁ�{�z�n�l�a�U�K�U�Z�a�k�n�w�z�z�H�T�a�g�m�w�}�~�~�z�m�a�`�T�Q�K�D�>�B�H�r�~�������ĺɺѺֺ�ֺƺ������r�e�`�k�r�Ľݽ���������ݽнĽ��������~���������
���#�&�+�+�#��
����������������
�
�f�s�~�y�w�t�p�j�f�Z�M�F�@�A�F�J�M�R�Z�f����������������������������������������������������������������s�e�l�s�x������5�A�N�O�Z�_�f�Z�N�A�@�5�4�4�5�5�5�5�5�5�������������!�/�6�4�"�	�����������������(�5�A�Z�]�]�V�N�A�5���������!�(���������������������������������~�y�y��àìù����������������ìÓ�z�p�n�r�{ÈàD�D�D�D�D�D�D�D�D�D�D�D�D�D}DmDjDnD{D�D��������׼ڼռʼ�������f�U�M�F�K�V�f�r�����*�6�C�J�O�S�O�O�C�6�*��������:�F�S�_�l�u�r�l�_�Z�S�R�F�:�8�2�:�:�:�:�����ûлڻлŻû������������������������g�t�t�g�e�g�g�g�g�g�g�g�g�g�g�g�{ŇŔŗŠŞŗŔōŇ�{�n�b�U�O�U�_�h�n�{ R / ( J B ? * A + e ? D M P D P � 8 B < p 9 A � J - / D 6 ( ) 5 + = 9 Y K � a = j ! k X X X b X \ * J + 1 5 K - D    w  �  �  �  @    �  Q  �  �  �  �  �  V  "  I  �  �  �     x  f  �  �  �    �  �  I  �  �  �  "  }  �  e  �  �  Q  �  /  1  *  t  �  �  ^  �  <    �  �  &  �  i  #  ȼ�h��`B�o<o<ě�=�P<�h=@�<�/<�9X<ě�<ě�=+=8Q�=49X<��
<�=]/=,1='�=�{=<j=o<�=C�=\)=�E�>�R=+=]/=���=m�h=@�=@�=#�
=m�h=��
=D��=��P=��=�t�=y�#=��w=]/=}�=�C�=�/=���=��P>n�>aG�>�J=�>   >�>J>5?}B	"�BW�B#Bk�B��BR.B	��B��Bu�B6NB1wB�PB�GB��B"0QB#&-Bu`B!PB��B�B�{B�Bh�B ��B#�ZB�WB^�B�BB��B
�B�$Bt�B�5Bw*B"�B-@Be�A���B��B"w�A�_)B�6B�B��B�aB�IB
o�B*�BS BܴBN(B2�Bn�BBB	3Bm�B	<�B��B?�BB�B°B��B	�B�|BYrBG�B@�B��B��B��B"$�B#AZB>�B?B�RB:�BS�BĻBCB ��B#��B~,B�~B��B��BAB	��B\uB��B�AB�yB!��B<BJ�A�[B?�B"_OA���B�0BFB��B�3B9�B
@KB*�8B��B��B@�B>YB�}B?�B:SB@�A�=cA/�A�C�A�^?��AFANA�:�A��4C�p�A��A�A�	A��A�E,@�)w@�
�Aθ6A�^�A�ҧA�Al#�A�8�@{�p@q�Q?Ż7A�۸B�WA�hwA���A�z�A�]?@��"A���A~�@A=��A��A��A�S�A�j1@p8A&�xA���A@��AH!iAG5�A��A�o8A��A 8�A̽vC��}@�A�B F@�JT@��A��^A�B�A���A.u�A�}A��?�-AE2A�
tA�dQC�nVA�x�A�BA���A�f�A�B�@���@��vA΀
A�r�A�^�A�}aAkH�A�w<@�@sȎ?�h�A�nVB��A�qA�z�A��A�~�@�$�A���AA=� A���A��qAǔ�A��x@�'A*��A� �A?AHQ�AE�A�#A��A��BAB"A��C���@�feA���@�K@��A��A�f         	   
      $      %      
                        #         ?            	   	   ?   �         /            
      .         6                  
   0         E   �   �               5               #   3      +                  -                     /                  5   1         %         !         /         #                     1         !      *            
                     )      +                  -                     %                  5            !                                                1                           
   Nh��NNTNaN�^Nv�PP��N�=�P#��N��Ny+�N��N]@�N�blP8`=O,�9N��Nm��O��NV8�N\�O�o�Nae�Ntr�N�jIN��LN��P��O�r�N2��OAO��O��N�OXR�N��vN�d�O.�mM��mO.�DO�)�Ok`�OvcO][�N1�zO�WN�-DP'k�O��N�"�O���O�BOH��N�&[N��rNJިNk�OT(  �  �  c  $  �  �  S  Y  �  �      �  ~  �  o  �  �  ]  �  	E  n    �    V  �  �  �  c  !  �    �  z  �  =  
)  �  �  �    k  �  �  �    `  �  
?  �  E  �  f  �  ,  L�\)��󶼃o��o<T��<e`B<#�
<e`B<D��<T��<e`B<e`B<u<�o<�9X<�o<�C�<�`B<���<�/=#�
=o<���<���<���<�/<�/=�t�<�h=t�=#�
=��=C�=C�<��=�w=m�h=,1=T��=Y�=D��=L��=T��=T��=T��=q��=y�#=�o=�o=�1>\)>333=�^5=�
==�`B=>   WVQ[dgotwtjgf[WWWWWW56BOOWUOB?;655555555��������������������@BGMNP[\__][NKGB@@@@ttu���������|ttttttt<BO[t���������t[I@6<eeegqtv������tlgeeee����������������������������������������DOR[hrtkh[ZODDDDDDDDyw{���������yyyyyyyy��
"!
���������DGLNU[giqmg][NDDDDDD�����
#/8HNI</#��������������������������������������������������������������

��������������������������429<DHUUVUH<44444444����
#/5EJH</#����������������������ffhkt������thffffff�������������������������

	�������������������������������)K[_[]XN5���������).47<<7)���������������������������
 
������TWX\]dg���������tb[T46;BFOTY[bhjhXOB<864<==BHUZ`a]UH<<<<<<<<����&585/*)������������������������������������������32245>BN[^dhjg[TNB53��������������������		"/;HONHB;/$"		

)6BOSZWSOG@6)��������������������"/;CFB>;0/" ����)58>:5)������������������������������
�����������������������������������).+�����qprrst�����������|tq������������������������������������������������


��������������������������	 
"#*+)##
		tpoosz���������ztttt`ZUWajnnnnja````````��������������������������
"##
���������������������������������������������������������������������������������������ĿĿĿ������������������*�,�*�!������������������L�V�Y�a�e�g�e�Y�L�A�@�;�@�J�L�L�L�L�L�L�������þ�����������e�X�Y�V�Q�V�f�s�����a�n�z�ÇÉÇÂ�z�n�a�U�M�J�U�[�a�a�a�a�g�s���������������{�g�Z�A�(���!�3�I�gE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��H�P�U�[�\�U�H�<�;�;�<�?�H�H�H�H�H�H�H�H�/�<�H�J�N�H�D�<�/�(�#�)�/�/�/�/�/�/�/�/�����������������������������������������U�b�n�r�{��|�{�n�b�\�U�T�R�U�U�U�U�U�U�/�H�T�a�o�m�e�^�T�/���������� ���"�/����'�0�4�:�5�'��������������û̻л׻лû������������ûûûûûûû�����������������ùóïöùÿ�������������U�a�n�n�q�t�r�n�a�U�H�<�5�/�)�/�6�<�H�U�)�6�?�B�C�B�6�*�)�����"�)�)�)�)�)�)�����������������������������������뿒���������������y�T�G�8�0�2�7�C�T�`�m�����������������������������������������޻-�:�=�F�Q�L�F�<�:�7�/�-�)�,�-�-�-�-�-�-���!�-�2�-�)�-�:�<�A�;�:�9�/�!�����3�@�L�Y�b�e�i�e�Y�L�@�9�3�,�3�3�3�3�3�3��*�2�6�C�G�O�O�C�?�6�*�!�������ƁƧ���������������ƚ�u�c�S�U�T�\�uƁ����#�0�=�C�>�7�0�#��������������������)�5�B�E�I�B�5�)�!�&�)�)�)�)�)�)�)�)�)�)�������������������������������������	��"�/�1�8�3�/��	�������������������������� ����ܻػл̻˻лܻ������)�2�2�)����������������ݿ�������������ݿѿпǿĿÿ˿տݾA�M�Z�f�j�j�f�Z�Z�Z�M�A�A�<�A�A�A�A�A�Aù��������������ùìàÓÒÌÓàèìôù�O�[�h�t�vāąĂ�y�t�h�[�O�F�F�B�@�E�I�O�z�|�|�z�q�n�d�a�]�_�a�m�n�z�z�z�z�z�z�z�H�T�a�a�m�t�y�z�v�m�a�X�T�Q�J�H�G�D�G�H�r�~���������˺Ժ������������~�r�g�c�n�r�Ľݽ���������ݽнĽ��������~���������
���#�&�+�+�#��
����������������
�
�f�s�|�y�v�t�o�i�f�Z�M�H�A�B�F�J�M�S�Z�f����������������������������������������������������������������s�e�l�s�x������5�A�N�O�Z�_�f�Z�N�A�@�5�4�4�5�5�5�5�5�5�������������!�/�6�4�"�	�����������������(�5�A�N�Z�[�\�Z�T�N�A�5�(������#�(�����������������������������������|�~��ìù����������������ìÇ�z�y�yÂÎÓàìD�D�D�D�D�D�D�D�D�D�D�D�D�D�D~D{D{D{D�D��r�����������������������r�f�e�`�a�f�r���*�6�C�J�O�S�O�O�C�6�*��������:�F�S�_�l�u�r�l�_�Z�S�R�F�:�8�2�:�:�:�:�����ûлڻлŻû������������������������g�t�t�g�e�g�g�g�g�g�g�g�g�g�g�g�{ŇŔŗŠŞŗŔōŇ�{�n�b�U�O�U�_�h�n�{ R Q ( J 6 N * = + e < D M P ; P � 1 D 8 l 5 A � J - / / 6 ' 0 ) . + 9 _ & � [ ; j ! k X X X b = ] # (  1 5 K - D    w  v  �  �  �  �  �  �  �  �  �  �  �  V  v  I  �  _  x  t  ;  v  �  �  �    �  �  I  7  �  J  �  �  �  �  v  �  �  r  /  1    t  �  �  ^  L    �  ?  �  &  �  i  #  �  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  B�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  c  ^  Z  R  H  :  #    �  �  �  �  h  G  &  �  �  �  o  >  $          �  �  �  �  �  n  J    �  �  �  �  \  2    �  �  �  �  �  �  C  T  f  w  �  y  V  #  �  �  p  )  �    6  G  Q  a  o  {  �  �  y  f  J  ,    �  �  Z  -  �  �   �  S  M  D  7  ,      �  �  �  �  Y  0    �  g  �  n  �  �  F  T  Y  R  C  .    �  �  �  u  O  &  �  �  l  �  v  �    �  �  �  �  �  �  �  w  d  O  7      �  �  �  �  w  =  �  �  �  �  �  �  �  �  �  �  �    x  o  g  `  \  c  q  �    �  �    �  �  �  �  �  �  �  u  R  .    �  �  p  ,  �  �    �  �  ~  y  m  a  U  G  6  "    +  l  p  X  ?  %    �  �  �  �  k  P  )  �  �  �  Z    �  �  N    �  Y  �  X  �  ~  s  _  G  ,    �  �  �  �  �  c  =    �  �  n  H  &    r  �  �  �  �  �  �  �  �  t  C    �  �  �  �  �  a  F  �  o  a  S  D  6  (         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  i  `  q  �  �  �  �  �  {  {  {  y  N  {  �  k  �  �  �  �  �  �  �  v  U  %  �  �  L  �  J  �  �    '  P  �    <  e  �  �       7  Q  k  �  �  �  �  �  �  �  M  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  (  �  �  i  y  �  	  	'  	<  	D  	D  	<  	#  	   �  �  3  �  :  �  �  �  �    ^  �  �    @  X  i  n  l  d  X  L  <  '    �  �  g    �                  	     �  �  �  �  �  �  v  B    �  �  �  �  �  �  �  �  �  �  �  �  �  y  _  E  "  �  �  �  �      �  �  �  �  �  �  r  [  D  .    �  �  �  �  j  H  &  V  I  <  *      �  �  �  �  �  r  Z  ?  $    �  �  �  \  �  �  �  �  �  �  �  �  r  8  �  �  �  q  =  �  x  �  C    
�  �    g  �  �  �  �  �  �  �  \    �  
�  
>  	F  �  4  �  �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �    @  X  `  c  b  ]  T  C  )    �  �  o  4  �  �  ;  �  �  �  �           �  �  �  �  �  v  F    �  ~    H  3  �  �  �  �  �  �  �  �  �  �  �  �  �  f  @    �  �  �  D  �  �  �           �  �  �  �  �  �  �  �  �  f  H  #  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  t  c  L  (  �  �  �  z  l  ^  K  9  )    	  �  �  �  �  �  �  j  U  H  D  �  �  _  �  �  �  �  �  �  �  �  j  K  0      
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  7  ;  .    �  ~    �  �  �  P  a  �  �  �  �  	#  	P  	|  	�  	�  
  
^  
�  �  �  �  �  �  �  
  �  �  �  �  �  �  �  �  �  �  �  �  ~  ;  �  y  �  v  *  <  L  �  �  �  �  X  %  �  �  L    �  ;  �  g  �  �  �  0  i  �  �  �  }  n  W  :    �  �  �  ~  V  +    �  �  s  D  �      �  �  �  �  �  �  �  l  R  4    �  �  �  }  J  �  �  g  c  W  L  6  0  0    �  �  �  M    �  k    �  X  �    �  �  �  �  �  �  �  �  �  �    |  z  w  u  r  p  m  k  h  �  �  �  �  �  �  �  �  u  d  R  @  *    �  �  �  �  �  �  �  �  �  �  �  l  U  7    �  �  �  v  D    �  �  t  B      
  �  �  �  �  c  1  �  �  r  -     �  �  @  �  r    p  4  P  ^  Z  T  K  <  ,      �  �  �  �  U    �  �  C   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  c  J  1  3  A  
  
6  
=  
>  
5  
'  
  
  	�  	�  	�  	�  	K  �  o  �  5  v  �  Y  @  �    �    H  j  }    i  "  �  �    �  �  �    U  	C  �  �    _  �    z  �    9  C  !  �    ?  a  N  
�  Q  �  �  �  e  B    �  �  �  �  N    �  p    �    D  i    �  f  B    �  �  �  p  @    �  �  z  G    �  �  �  �  �  b  �  �  s  5  �  �  f     �  �  @  �  �  I  �  �  *  �  S  �  ,    �  �  �  �  �  �  ]  )  �  �  �  �  {  [  o  �  �  �  L  �  �  _    �  �  :  
�  
�  
  	�  	.  �    .  ?  I  �  
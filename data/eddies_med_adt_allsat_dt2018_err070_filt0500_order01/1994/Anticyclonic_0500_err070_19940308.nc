CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���S���       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mƀ"   max       P`(�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =�t�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�p��
>     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vz�\(��     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @Q�           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�V�           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       >�       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�Ie   max       B,��       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,��       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?=U�   max       C���       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O�   max       C���       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          m       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mƀ"   max       PVto       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?� ě��T   max       ?�.��2�X       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       =���       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>ٙ����   max       @E�
=p��     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vz�\(��     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q�           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?	   max         ?	       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��S&�   max       ?�)�y��     �  Y|            	   N                        
      O      	               (                                 
                                 '      l         6   %               &   	         	               0      JN.�N�Nf��N�]�P�ON*��Nk�nN
��N{@�Oh��N���M��iNbq�N��PC;N�?O{�Oo"*N��JN<��N+ׄP`(�N-�+N��
Na�O���O��OC%NɭRN�pOp��N��@O��.O*�iNk+�N�y{O�>�O9�N-QRN�#�O��N6	WNWO��N�mO�j:OfpvO~BlP�P��N�QO�&O�� N�O�O�f�OT�N���NEe�Mƀ"N�LoN�w�N::�O�O��HNnr�O�E,�t���`B��/��C��o�o�ě����
��o�D���D��;o;�o;��
;ě�<#�
<#�
<#�
<49X<D��<T��<T��<T��<e`B<u<u<u<u<�o<�o<�C�<�C�<�t�<���<���<��
<��
<��
<��
<�1<�9X<�j<ě�<�/<�`B<�h<�h<�=+=+=\)=�w=#�
=#�
=,1=<j=<j=<j=P�`=e`B=q��=y�#=y�#=�%=�\)=�t�rt����������xtrrrrrr_afgqtx}�����}tojg__��������������������QOQ[gqtytrg[QQQQQQQQa_bi��������������ka��������������������dhjst�����thdddddddd����������������������������������������������������������������������������������������������������>@BGOY[`[YOB>>>>>>>>��������������������	  /;H[]ZM;"	�),*)()58>A<5.)#����
#/3<BB<7#
��')+5BEMNQNCB:5,)''''f]]httvzutthffffffff25BN[ghg[PNB>5222222�������"$/*�����TR[^hjljh[TTTTTTTTTT���������������������������������������������"$% ����sklttvz�����������ts##0<IUYbbUTLB<0-$#$),257;<<95+)������������������������)9CB96#�������������������������������������������������������������������������������������
#/3/'#
�����=>CMUanwz�����znaUI=��������������������Z[ht���thg[ZZZZZZZZ���
##+/#
��������������	��������������������������������������������"#)/<HS\[ULU^U<+"����������������������������

�������������

����������������������������������������5BILPMB5)��602;HT]XTLH;66666666������ 	���dggcdg����������xtgd�����������������������������������������	���yv��������������yyyy#%+010#abmn{|{nbaaaaaaaaaaZYXanz{}~|znaaZZZZZZ&)2699863+) "%/0/*"��������������z|�����������������868<CHLTRMH<88888888z������������������z�����������������������������������������zÇÓÕàáàÓÇÆ�z�n�d�a�U�a�n�q�z�z����	����������������������������������������������������������������B�O�[�h�x�z�w�s�[�A�>�)�#�%�����6�B²¿��������¿¹²«²µ²«²²²²²²�L�P�Y�e�h�q�e�Y�L�I�H�J�L�L�L�L�L�L�L�L�'�3�4�4�3�'�����'�'�'�'�'�'�'�'�'�'��������	��
�	������������������������������������������������������������������������������������������������������������������������������������������������y���������������~�y�l�o�y�y�y�y�y�y�y�yìùùý��ùùìàÓÒÓÖÝàçìììì��������;�H�a�s�����r�H�;�"�����������׾Z�f�g�h�g�f�^�Z�M�I�G�M�N�X�Z�Z�Z�Z�Z�Z�׾����	��	�� �����׾ϾʾǾʾо׾׿`�m�y���������}�y�m�`�T�H�G�D�G�L�T�Z�`�(�-�4�9�6�4�(�!��������"�(�(�(�(���ʾ׾߾׾־ʾ��������������������������;�?�@�C�?�?�;�:�9�.�-�,�.�7�;�;�;�;�;�;�B�c�t�}�|�t�[�)��������������������)�B�-�:�E�F�K�F�:�-�%�"�-�-�-�-�-�-�-�-�-�-�"�/�4�;�@�>�;�/�"��	���"�"�"�"�"�"�"�*�-�6�=�;�6�5�*����$�*�*�*�*�*�*�*�*�\�h�uƁƎƒƛƞƚƎƁ�u�h�c�O�M�E�G�N�\�A�M�Z�f�k�s���������s�f�Z�M�D�A�:�@�A�r�����������������������r�f�_�X�Y�f�r��)�5�5�B�J�N�[�e�[�N�B�5�)�"������ܹ������$����� �����ܹعֹܹ����	��"�/�2�0�,�"��	��������������������"�*�.�4�6�.�)�"����	�
�������(�4�N�_�f�f�Z�M�A�4�(���������(�ѿݿ��������ݿѿĿ��������������ĿοѺ�����!�"�!�����������������������5�7�5�-�(�(�&�&���������(�)�5�5�5�A�N�Z�f�m�j�Z�V�A�5�(������ �(�5����������������������������������ÿ�ſݿ߿��ݿٿѿƿ˿пѿܿݿݿݿݿݿݿݿݾ(�(�)�4�7�A�D�A�6�4�(������&�'�(�(�	��/�;�H�Q�T�V�U�N�H�;���	���������	����!����������������FFFF$F.F$FFE�E�E�FFFFFFFFF��������
���������������ùíò���Ҽ����ʼּؼ��ּʼ���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DtDvD{D�D������ʾ׾�����Ҿ������������}�����������нݽ��������ݽн�������������������ĚĦĿ������������ĿĦčā�t�i�b�g�kāĚ�#�<�I�b�q�v�l�a�U�I�#�
����������� ���#čĚĦĬıĬĦĚĔčĆăčččččččč�-�:�F�S�_�i�l�m�_�S�F�:�-�!���	���-�����������������������s�Z�J�C�D�N�W�b���`�l�y�����������������y�s�l�f�c�`�Z�`�`�������������������������������w�u�v�{���׾����	��"�$�'�"��	�����׾;̾׾��<�H�O�U�V�U�Q�H�?�<�/�+�'�)�/�6�<�<�<�<���!�,�.�4�.�!����������������������������������������������������ûͻʻĻû��������������������������r�~�������������������~�v�r�e�]�e�f�r�r���
���#�%�#����
�����������������¿����������������������������¿¿¼¾¿ì������������������ùìçàÔÎÒÓàìEEEE&E*E,E*EEEE EEEEEEEEE���������ּ����������ּ��������������� e L [ / B T a I h H I g : : K c 7 G - g � v 9 ; F > _ @ i K 8 N 1 O , i 3 P t \ 7 K { A (  q ` K  5 x _ 4  c E G q 4 @ I N " , ]  �  �  �  �  �  D  �  1  �  �  �  .  o  �  �  �  -  �  �  D  �  �  C  �  8  (  H  �  '    �  �  �  �  w    .  �  �    $  e  b  (  �  S  2  @  �  �  �  �  �  �  �  a  �  w  /  �  �  n  b  �    �+���
��j�o=�O߻o;��
$�  :�o<o<o;�o<T��<�o=�1<u<���<��<T��<u<u=T��<���<�C�<�C�<�`B<�1<�`B<�j<ě�<�`B<�1<�`B<�/<���<���=49X=0 �<�j<���=��<���<�`B=�+=�w>I�=<j='�=� �=�\)=#�
=u=Y�=T��=��
=]/=}�=D��=u=�\)=��=�+=���=�S�=�->�B
TB	��BFB	hBaB 0�B]�B!ymB�B�MB!ۃB"=�B��B!�SA�T�B�B��BqBz�B�B 7B��Bs+B�B��BK�BڞB&S�B�VB DB��B �;B
�B��B 8�B5�Bz�B
�B��BF�B�B�kBH�B��B"��B��Bg�B �AB+NBw�A�K�B�5B
�B,��BM�ByqB�,B%�B(F:B�[B-A�IeB�PB%�B�B�IB
k�B	��B��B	9~B9�B�'B�B!H�B]�B�YB"�B"?oB��B!��A�\OB*�B��BGnBl�B�mB�mB:�BOzB�MB�iBC�B��B&?�B8�B�8B��B ��B�@BM�B AnB��B�B�B�BsB�B� BH~B�HB"��B�uBBB �)B7�B�eA�~�B�=B
?B,��B�
B�B��B%!SB(@KB�zB?�A�~�B�iB@OBAB�A�e:A��A��tA��4A�� A�,�?�!�?�(A��^A�X@E�@���A��Å�A��$A>�`AV?:Ai�MA6#kAP��AdOvA�{�@y�A�� A���B��A@��@�JUA�޼?=U�A��A]�	A8J�A{��@^O�A�A�A��AІ�A|)A7�A�/A�h�C���A�٣@��C���AN�mA&�6A���A�BA߱:@|�MA�лATA��wAX��AÏ�A��@T�@�V�@��A���A�� A͟C�l�@�Q�A�m�AȐ�A��A��A؀uA�[?�(?��A�rpA�� @v@���A�A�cUA�j�A?bAT��Ai>A7\�AQ8Ac=�A�u;@{�/A�0�A��\BxgAA�@���A���?O�A�SSA^T�A:��A|��@_M�A��rA�y�A��A|<A7��A���A�|C���A���@���C���AQkA'!�A��yA��Aߊ
@r�A��!ALA��AXonA�{&AX@T�@��@\rA�[�A��A��C�t�A �n      	      
   N                        
      O      
               (                        	            	                              (      m         6   &               '   	         
               1      K               %                              -                     ;                                 #                                                '   '               !                                 &                                                                  ;                                                                                                   !                                 N.�N�Nf��N9rFOy�ZN*��Nk�nN
��N-�KO)ޮN���M��iNbq�Ns�O���N*2�O{�Oo"*N��JN<��N+ׄPVtoN-�+N��
Na�N���O��OC%NɭRN�pOP��N��@O�v(N��Nk+�N�y{O}U�N��N-QRNw�5O-��N6	WNWN��N�qN챲OfpvO~BlOY��O�[*N�QO�&O�� N��fO�f�OT�Nt��NEe�Mƀ"N�LoN�1N::�O�O^j$Nnr�O���  �  ^  	    	H  {  �  �  �  �  �  �  �  �  	  N  �  �  ;  m  �  �  l  W  �  �  �    %  B  W  �  �  4  R  U  -    �  �  �  �  �  �  u  �  v  5  e  +  a  �  "  I  �  �  +  �  �  �  J  �  z  �  �  ��t���`B��/�u<�1�o�ě����
�D����o�D��;o;�o;�`B=t�<49X<#�
<#�
<49X<D��<T��<e`B<T��<e`B<u<�1<u<u<�o<�o<�t�<�C�<���<�9X<���<��
<�9X<�/<��
<�9X<���<�j<ě�=49X<�h=��<�h<�=e`B=49X=\)=�w=#�
=0 �=,1=<j=L��=<j=P�`=e`B=u=y�#=y�#=�7L=�\)=���rt����������xtrrrrrr_afgqtx}�����}tojg__��������������������RV[gmtutlg_[RRRRRRRRpmqt~�������������{p��������������������dhjst�����thdddddddd����������������������������������������������������������������������������������������������������>@BGOY[`[YOB>>>>>>>>��������������������"/5;JKKC;/")+)#()58>A<5.)#����
#/3<BB<7#
��')+5BEMNQNCB:5,)''''f]]httvzutthffffffff25BN[ghg[PNB>5222222�����!#-(������TR[^hjljh[TTTTTTTTTT��������������������������������������������������sklttvz�����������ts##0<IUYbbUTLB<0-$#$),257;<<95+)����������������������� 
)5A?76)�����������������������������������������������������������������������������������
#/3/'#
�����ABFPUanuz�����znaMHA��������������������Z[ht���thg[ZZZZZZZZ
!#)*#
����������
��������������������������������������������)'(//<DHPMHD</))))))���������������������������

	 ��������������

����������������������������������������������)5BIKFB5)�602;HT]XTLH;66666666������ 	���dggcdg����������xtgd�����������������������������������������	�����������������������#%+010#abmn{|{nbaaaaaaaaaaZYXanz{}~|znaaZZZZZZ').679862*)& "%/0/*"��������������~������������������868<CHLTRMH<88888888�������������������������������������������������������������zÇÓÕàáàÓÇÆ�z�n�d�a�U�a�n�q�z�z����	����������������������������������������������������������������B�O�[�_�h�i�l�k�h�f�[�O�B�6�)�"�"�)�6�B²¿��������¿¹²«²µ²«²²²²²²�L�P�Y�e�h�q�e�Y�L�I�H�J�L�L�L�L�L�L�L�L�'�3�4�4�3�'�����'�'�'�'�'�'�'�'�'�'���������	�	�	�	�	�����������������������������������������������������������������������������������������������������������������������������������������������y���������������~�y�l�o�y�y�y�y�y�y�y�yìùû��ù÷ìàÜØàáìììììììì��"�/�;�H�T�a�f�f�a�Y�;�/�"�	���������Z�f�g�f�f�\�Z�Q�M�K�M�Q�Z�Z�Z�Z�Z�Z�Z�Z�׾����	��	�� �����׾ϾʾǾʾо׾׿`�m�y���������}�y�m�`�T�H�G�D�G�L�T�Z�`�(�-�4�9�6�4�(�!��������"�(�(�(�(���ʾ׾߾׾־ʾ��������������������������;�?�@�C�?�?�;�:�9�.�-�,�.�7�;�;�;�;�;�;�B�a�t�s�[�)�����������������������)�B�-�:�E�F�K�F�:�-�%�"�-�-�-�-�-�-�-�-�-�-�"�/�4�;�@�>�;�/�"��	���"�"�"�"�"�"�"�*�-�6�=�;�6�5�*����$�*�*�*�*�*�*�*�*�\�h�uƁƂƆƊƄƁ�u�s�h�a�\�Y�X�\�\�\�\�A�M�Z�f�k�s���������s�f�Z�M�D�A�:�@�A�r�����������������������r�f�_�X�Y�f�r��)�5�5�B�J�N�[�e�[�N�B�5�)�"������ܹ������$����� �����ܹعֹܹ����	��"�(�/�1�/�*�"��	������������������"�*�.�4�6�.�)�"����	�
��������(�4�M�[�c�Z�U�M�A�6�1�(��������ݿݿ����ݿѿĿ��¿Ŀѿݿݿݿݿݿݿݺ�����!�"�!�����������������������5�7�5�-�(�(�&�&���������(�)�5�5�5�A�N�Z�b�j�g�Z�T�A�5�(������(�-�5���������������������������������������ҿݿ߿��ݿٿѿƿ˿пѿܿݿݿݿݿݿݿݿݾ�(�/�4�A�A�A�4�2�(�����������	��"�/�;�@�H�K�H�G�;�/�)�"��	������	����!����������������FFFF$F.F$FFE�E�E�FFFFFFFFF���������	�������������������������޼����ʼּּ�ۼּʼ���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D������ʾ׾�����Ҿ������������}�����������нݽ��������ݽн�������������������āčĚĦĳ��������ĿĳĦĚčā�y�t�v�ā��#�0�<�I�N�W�]�[�U�I�0�#�
�����
��čĚĦĬıĬĦĚĔčĆăčččččččč�-�:�F�S�_�i�l�m�_�S�F�:�-�!���	���-�����������������������s�Z�J�C�D�N�W�b���l�y���������������y�l�k�g�d�l�l�l�l�l�l�������������������������������w�u�v�{���׾����	��"�$�'�"��	�����׾;̾׾��/�<�H�Q�M�H�<�<�/�.�)�,�/�/�/�/�/�/�/�/���!�,�.�4�.�!����������������������������������������������������ûͻʻĻû��������������������������r�~�������������������~�x�r�e�c�e�h�r�r���
���#�%�#����
�����������������¿����������������������������¿¿¼¾¿ù����������������ùìàÖÓÐÓÕàéùEEEE&E*E,E*EEEE EEEEEEEEE�������ʼּ��������ּʼ������������� e L [ 6 . T a I l L I g : < @ \ 7 G - g � z 9 ; F ; _ @ i K . N 3 . , i 3 0 t ; / K { " (  q ` 1 # 5 x _ 6  c E G q 4 @ I N # , [  �  �  �  ]  �  D  �  1  �  �  �  .  o  �  q  h  -  �  �  D  �  L  C  �  8  	  H  �  '    �  �  E  �  w      �  �  �  p  e  b  �  �  �  2  @  �  �  �  �  �  �  �  a  {  w  /  �  �  n  b  �    [  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  ?	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  U  M  C  7  +         �  �  �  �  �  �  }  i  B  �  �  	      �  �  �  �  �  �  �  �  �  �       
    �  �  �  �  �  	              �  �  �  �  �  {  \  9    �  �  �    N  �  	  	=  	F  	H  	D  	7  	  �  �  J  �  b  �  �  O    {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      "  %      �  �  �  �  w  W  4    �  �  �  �  �  �  �  �  �  �  x  Y  ;    �  �  �  �  ]  4     �  Y  f  r    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  N  "  �  �  <   �  �  �  �  �  �  �  �  �  �  �  y  _  E  *    �  �  �  A  �  �  �  �  �  �  �  �  �  ~  z  r  h  ]  S  H  >  3  )      �  �  �  �  �  �  �  �  q  _  N  <  ,        �  �  �  �  b  p  }  �  �  �  p  ]  D  *    �  �  �  s  �  e  A    �  �  �  �  2  �  �  	  	  	  	  �  �  �  D  �  �  �  ,  =  �  .  6  ?  G  J  7  %    �  �  �  �  �  s  U  7     �   �   �  �  �  �  �  �  �  �  w  d  O  9       �  �  �  x  X  6    �  �  �  �  �  �  g  A    �  �  {  C  
  �  �  X  @  �  �  ;  =  @  C  F  I  L  O  R  T  U  S  Q  P  N  L  J  I  G  E  m  g  a  [  U  N  H  ?  4  )      	   �   �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  \  k  x  r  S  /    �  �  j    �    Y  l  m  n  o  n  m  l  g  `  Y  M  >  .    �  �  �  k  -  �  W  O  H  @  8  1  )         �   �   �   �   �   �   �   �   o   [  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  T  ;  '    �  �  �  �  �  �  �  �  �  �  |  �  �  �  �  �  �  �  �  �  �    n  ^  Q  S  c  x  w  o  c  V  H  :  ,         �  �  {  %        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    B  <  6  /  '             �  �  �  �  �  s  )   �   �   w  D  O  V  O  F  <  0  "      �  �  �  �  �  �  i  H  )  	  �  �  �  �  �  �  z  q  h  ^  S  E  7  )         �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  c  Q  @  {  �  �      *  /  3  4  4  +  !    	  �  �  �  �  �  �  u  I  R  K  C  <  -      �  �  �  �  �  �  �  m  W  B  -      U  R  P  M  G  7  &       �  �  �  �  d  C  "     �   �   �  ,  -  +  &      �  �  �  �  �  x  X  9    �  �  y  D  @  �  �  �              �  �  �  �  �  h  <  �  �  ]  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  u  {  �  �  �  �  �  �  �  �  �  �  �  �  �  l  =    �  �  .  �  �  �  �  �  �  �  v  h  Z  L  >  0  %           �   �  �  �  �  �  {  l  \  K  :  )      �  �  �  �  �  �  �  �  �  �  �  6  �  �  �  �  �  �  �  �  �  Y    �    }  �  <  l  q  t  q  j  `  U  I  9  *      �  �  �  `  )  �  �  �  2  �  3  �  �  J  �  �  �  �  �  �  P  �  �  &  K  F  	�  �  v  h  \  P  ?  )    �  �  �  �  Z  -    �  �  �  �  �  M  5  &        �  �  �  �  �  �  �  �  �  �    x  l    �  L  �  �  �  �    8  T  d  `  N  %  �  �  �  .  �  G  �  P  �        "  )  *  #    �  �  �  �  \    �  g  �  �  �  a  Z  R  J  B  ;  5  .  '          #  1  ?  Z  w  �  �  �  �  �  �    q  a  L  3    �  �  N  �  �  $  �  2  �  �  "      �  �  �  �  �  c  =    �  �  �  �  �  e  (   �   �  E  C  D  H  I  H  C  >  8  ,      �  �  �  �  s  Q    �  �  �  o  P    �  �  q  W  ;      �  �  s    �  8  �  �  �  �  �  �  �  �  �  {  k  [  L  >  /      �  �  �  �  {  �      "  (  (      �  �  �  |  M    �  �  4  �  �  D  �  �  �  �  �  �  �  �  �  �  �  ~  v  o  g  _  W  O  G  ?  �  �  �  �  o  C    �  �  �  T  "  �  �  �  N    �  �  i  �  v  d  O  :  $  
  �  �  �  �  {  a  B  "    �  �  �  �  B  H  E  =  .      �  �  �  d  :    �  �  2  �  �     �  �  n  U  =  %    �  �  �  �  �  �  �  �  z  k  H  !  �  �  z  v  i  U  A  *    �  �  �  �  y  O  5    �    8  �  �  �  �  �  �  �  �  �  �  �  [  ,  �  �  #  �  �  �  �  A   �  �  �  �  �  Y  1    �  �  �  `  0  �  �  �  Y     �  �    
�  �  �  �  �  j  u  �  �  {  a    
�  
;  	�  	-  B    �  �
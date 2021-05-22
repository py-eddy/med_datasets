CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�l�C��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mݡ�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �H�9   max       =�^5      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @F��
=q     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?У�
=p    max       @vffffff     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @N�           x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��@          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �8Q�   max       >W
=      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�{�   max       B.3s      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�p   max       B.kQ      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?ib   max       C�O      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?MO�   max       C�2      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          +      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mݡ�   max       P�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ᰉ�(   max       ?��g��
      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �H�9   max       =�S�      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F��
=q     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @vffffff     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @N�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @�@          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Bv   max         Bv      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�-�qv   max       ?���rGF     �  T         	   /   
         5   "               �            4                        h               -      	   X   0         &   2                              
   $            �   �               6NwPO�-qNŝ�P��N�?N�y�O2��P|eROc$N#��NN�=O"��O4I�P���P�O	�JO�P+٬N��%N�2P�*NN�OS�O��NB}P{�OYm�N�M�N���NCg_O��JO��N��PP>��O�hN�zP �vO�EPvO��?N"-jO���N�mN�=�N]@�Mݡ�N�;'N�=�Nc�O��Ne+N�O���O��P)v�N��N���N�<�O&X O��f�H�9�u�e`B�T���T���#�
���
���
��o%   :�o;ě�<o<#�
<49X<T��<e`B<e`B<u<�o<���<���<���<��
<ě�<���<���<���<���<���<�`B<�=o=o=+=+=C�=C�=t�=t�=�P=#�
=#�
='�=49X=49X=8Q�=H�9=e`B=u=u=�o=�o=�+=�\)=���=��
=� �=�-=�^5��������������������
#0BQPRM=70#
�{{|�����������������������������������������������GHN[gtzttg][ZNGGGGGGgdhilt����������tjgg������->>:0
������! #/<HU[aijaXUH</#!�������������������������

	������������������������������)01>INNI<0#
!5gx��������t[5������������������������	"&/5/,#	��ZXURY[_gtxx|���tkg[Z /<HerutnaUH<7/������������������������

�����������0Bg}���gN5)#!��������������������WSSX\hqt�������{th[W�����

��������������������������������6MSL6����������������������������������������������������������������))6BIHB966-)))))))))�������%#����������")6<?<6)����
#)/,#
����������$5BHNLB5)'
������5:9;753,+���8=ABFN[ad][NLB888888�����������������������������

���������
#/HQVVN<4$#�� "#/<QY[ZSLH</*# NLNR[\ce[NNNNNNNNNNN�������%����sst�����}tssssssssss��������������������;;<<?EHRUWUPKH><;;;;<<HIUWUUOH@<<<<<<<<</.269<9BFJOW[c^[OB6/)+)'#ECEHUVWUQHEEEEEEEEEE:;=AHTamwz~��zmaTH<:�����������������������������������������������*6=6-*�����������
!" 
���������6CUF6)������������������������dehmstv�����ztmhdddd�����
##&&#!
��������������������
�����������
"! ����
�����
�������������������������S�_�l�x�����������������x�_�S�A�8�;�F�S�r�����������������������t�r�r�r�r�r�r�5�B�N�y�t�[�N�B�)�%�$�'�*�5����������������ùöùÿ���������������������������������������������������������zÇÓàìñõîìàÓÇ�z�s�n�m�j�n�x�z�A�M�b�f�X�}�{�s�M�(��߽Ľ��ݾ��(�+�A�������%�%���	�������������������²¿����¿²¦¦©²²²²²²²²²²�O�[�h�t��|�t�h�^�[�O�N�O�O�O�O�O�O�O�OÓàìù����������������ùìåäàÓÏÓ���������������������v�i�^�V�Y�f�r�|���6�[�tą�v�h�J�C�6�)���������������������������������������{�m�s�w�w�������;�H�K�T�a�b�h�m�m�l�a�X�T�H�;�4�.�/�8�;��������������žǾ������������������T�a�m�z����x�a�H�;�"��������������7�T��������
�
��������������������������޼�'�4�8�@�4�'������������������)�����ݿѿ˿ɿ¿������Ŀѿݿ��T�`�m�w�x�m�`�T�J�S�T�T�T�T�T�T�T�T�T�T�Z�f�s�������������������f�a�Z�U�N�Q�Z���������������������y�m�`�T�]�`�m�y�����;�H�K�Q�P�H�@�;�5�1�;�;�;�;�;�;�;�;�;�;�����������/�;�L�T�N�8�����������������������"�+�(�$���	������׾ԾϾ׾�Y�`�f�o�r�����r�g�f�Y�M�@�9�<�@�G�M�Y�A�N�X�Z�e�g�m�g�Z�N�A�?�5�<�A�A�A�A�A�A�ּ׼�߼ּʼ��������ʼռֺּּּּּּּ��'�2�,�/�9�:�'������ιʹ͹ܹ��.�;�Q�P�G�;�.�%��	��׾Ͼƾɾؾ����.������� �����ټּӼּ�������²¿������	���������²�q�c�^�g ²��#�=�H�N�T�I�<�0�#�
������������������U�b�n�p�{Łł�{�n�b�_�V�U�R�U�U�U�U�U�U�N�g�������������������s�Z�A�-�(�#�#�'�N�ѿݿ����"�-�&�#������ݿͿƿ¿ĿѾ����ʾ��%�2�:�8�,�"��	��پʾ��������4�A�M�Z�s�{��y�f�Z�M�A�4������(�4�I�V�a�b�e�b�V�I�=�I�I�I�I�I�I�I�I�I�I�I������!�1�9�9�-�!���������Ӻٺ����������������������������������������𼱼��ʼͼּּ޼޼ּʼ��������������������(�5�>�A�N�Z�f�Z�V�N�I�A�5�2�(�'�(�(�(�(�������������s�r�s�u�������������������������ɺѺֺ������Ӻɺƺ��������������	�������	� �������������	�	�	�	�	�B�N�[�g�^�[�N�E�B�9�B�B�B�B�B�B�B�B�B�BĳĿ����������������ĿĳĦĥĞĚĘęĦĳĚĦĨīĭĩĦĚĘđĘĘĚĚĚĚĚĚĚĚ��(�1�4�6�7�4�+�(��������������`�l�y�������~�����x�l�`�S�Q�L�K�O�S�]�`D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DpDsD|D�D��ܻ����������ܻɻ������������û��{ÀÇÍÓàáçèàÓÇÂ�z�v�p�r�v�z�{�'�4�:�@�M�N�M�D�@�4�'�%�����'�'�'�'ǈǔǡǬǭǶǭǬǡǠǔǈ�~�{�u�x�{ǅǈǈE�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EzE{E{E�E��I�G�K�M�Q�]�n�{ŇŋŗśśŘŒŇ�{�b�U�I = N 3 - { X # 5 - ( \ E / 7  W 5 9 B T U G F O [ E ) B 2 - N e > D , Z ] H _ 5 D , , I v h V I R # r  r . @ f 4 * S "  �  Z  �  V  m  �  w    �  3  �  �  �  :  3  Y  J  7  �  >  �  a  �  >  �  j  �    �  `  �  �  �  t  %  �  �       I  7         �  =  4  �  1     �    �  �    A  �    �  R�8Q�<o���
=C��D����o<�9X=@�<�;ě�;�`B<�C�<�9X>E��=t�<ě�<��
=�+<�`B<��
=0 �<���=\)=,1<�`B>o=#�
=0 �=�P=+=��=ix�=#�
=�=��T=0 �=�o=�t�=�{=�%='�=�\)=@�=u=@�=<j=e`B=P�`=��=ě�=�7L=���=�->W
=>Kƨ=�v�=�v�=�/=��>t�B�*B%�`B)��B1wB�BߴB
+�B$�XB��B �BY�B�zB%ށBbpB��A�{�B	X�B�)B�B#�9B�B!DBPB��BB8XB�ZB�GB�B�BW�By�B$�/B�B�9B#vB:KB�wB��B�B�4BQ�B��B~cB�B7?BmB�GB`WA��Bg0B�B.3sBIB}[B!2�BDjB��B9�B)5B��B%��B)�B@"B�fB�cB
?�B%,qB��B �B6!BT�B%��B@LB��A�pB	\cB�B�TB$'�B��B@5B��B�B
�B��B�]B��B��B�HB��B�<B$ұB��B�fB�uBDvB�B��B��B��B9�B�NB��BAIBM.BA�B�BB�A���B��BƭB.kQB=�B��B!��BGYB��B��B��A�қ@�P�@��A���A���A�'LA��A6�vA��A��4A�H(A��V@��A�yA��vA�&�AJ2_A��~A�K�@�m/A� AigAD�@Am��A��5A�oAYi�@�ҤA��A %?ibA[A�A�mRA�&A��/A���A���AU�@A<J�B�Q@\]A��V@�2A��dA���@4=uA�A0A��{A�M`A�_A4� Au#C��@���A��y@�&EB@�C�OA�]A�Kr@�Ƌ@��RA�-A΀A��AɑA8A҆�A�r+A���A�z3@�ʡA�t�A���A���AIjA��zA��y@��[A~}FAi^AC��An<A��CA��qAZ�@ك>A�~�@�rJ?MO�AY�PA/�A��cA�]�A��nA� BA���AY��A=
sB�*@a��A���@���A�v+A��V@, 9A�b�A���A��AߓzA6k*A�aC���@���AɈ�@���BD�C�2A�k,         
   0            5   #               �            5                        h               .      	   Y   1         '   2                              
   %            �   �                6      !      #            =                  A   #         +         +               9               '   #      -   !      )   #   -                                             !   +                                       !                  %            #         #               '               '   #      %   !      %      +                                                               NwPO=�fNA�VOl�N�?N�y�O�O�L�Nۀ,N#��NN�=O"��N�-AO�u�O��JN�6MO�O�QkN�.;N�2O��%NN�OS�N�M�NB}P	��OYm�N��GN���NCg_O�z�O��N��PO�,?O�FwN�zO�H�O{��P�On�BN"-jO{��N�mN���N]@�Mݡ�N�;'N�=�Nc�Ok�5Ne+N�O���OΧBO���N��N���N�<�O&X O��f  6  �  �  c  o  �  *  �    #  b  L  �  �  �  �    �  �  �    �  T  K  \  	�  �  �  �  n  �  �  S  
f  �  n  ;  �  �  �  �  �  (  W  �  �  �    �  =  �  �  "  {    �    �  �  
��H�9�t��49X;ě��T���#�
%   <�o;�`B%   :�o;ě�<e`B=�j<�C�<e`B<e`B<�`B<�C�<�o<�9X<���<���<ě�<ě�=u<���<�`B<�/<���<�h<��=o=]/=��=+=\)=0 �=�P=#�
=�P='�=#�
=@�=49X=49X=8Q�=H�9=e`B=}�=u=�o=�o=���=�S�=���=��
=� �=�-=�^5��������������������!#0<FIIE?<50#����������������������������������������������������GHN[gtzttg][ZNGGGGGGjhlnt{���������tjjjj�������
#03540#
��,(%)/<HKRSJH</,,,,,,�������������������������

	������������������������������! #*07<EG=<0+#!!!!!!(2?JN[goxyytg[NB5.*(���������������������� 	"/3/*"!	����ZXURY[_gtxx|���tkg[Z#/<WaimmkaUH<2% ������������������������

�����������)5Ngx|sgN:5)$!��������������������WSSX\hqt�������{th[W�����

������������������������������4<AB?6)���������������������������������������������������������������))6BIHB966-)))))))))�������#����������� )6;>;6)����
#)/,#
���������5AEGEA5)��������)/742'%���8=ABFN[ad][NLB888888������������������������������������������
#/HQUVM<3##��#(/<NTUSNKHE</(#NLNR[\ce[NNNNNNNNNNN������������sst�����}tssssssssss��������������������;;<<?EHRUWUPKH><;;;;<<HIUWUUOH@<<<<<<<<</.269<9BFJOW[c^[OB6/)+)'#ECEHUVWUQHEEEEEEEEEE><<>BHTamvz|}zxmaTH>�����������������������������������������������*6=6-*�����������
 
����������'++( �������������������������dehmstv�����ztmhdddd�����
##&&#!
��������������������
�����������
"! ����
�����
�������������������������x�������������������x�l�g�_�R�I�S�_�n�x�������������������}�|���������B�N�[�g�t�|�t�g�[�N�E�<�5�5�6�>�B����������������ùöùÿ���������������������������������������������������������zÇÓàéìïìéàÓÇ�z�w�p�n�z�z�z�z����(�4�M�[�d�g�`�Q�A�(������������������������������������������²¿����¿²¦¦©²²²²²²²²²²�O�[�h�t��|�t�h�^�[�O�N�O�O�O�O�O�O�O�OÓàìù����������������ùìåäàÓÏÓ�����������������r�r�i�r�|��������6�O�[�^�^�U�B�)�������������������������������������������������|�|�������;�H�T�a�a�g�l�j�a�Z�T�I�H�;�7�0�;�;�;�;��������������žǾ������������������T�a�m�s�v�s�a�U�H�;�/�"�������	�"�/�T����������������������������������������'�4�8�@�4�'��������������ݿ����$���	����ݿѿοſ������ĿѿݿT�`�m�w�x�m�`�T�J�S�T�T�T�T�T�T�T�T�T�T�Z�f�s�������������������f�a�Z�U�N�Q�Z�m�y���������������y�m�`�_�`�a�l�m�m�m�m�;�H�K�Q�P�H�@�;�5�1�;�;�;�;�;�;�;�;�;�;������/�?�B�?�7�"��	�����������������׾�����"�+�(�$���	������׾ԾϾ׾�M�Y�f�k�r�|�r�f�b�Y�M�@�<�=�@�J�M�M�M�M�A�N�V�Z�c�g�Z�N�A�@�7�>�A�A�A�A�A�A�A�A�ּ׼�߼ּʼ��������ʼռֺּּּּּּּ��'�1�,�.�8�9�'������۹Ϲ˹ιݹ��.�;�I�O�G�;�.�$��	��׾ѾǾʾپ����.������� �����ټּӼּ�������¿��������������¿²�|«²¿��#�0�<�E�K�M�L�I�0�#�
����������������U�b�n�p�{Łł�{�n�b�_�V�U�R�U�U�U�U�U�U�N�g�������������������s�Z�A�0�(�%�$�(�N�ݿ������ �#���������ݿͿѿҿݾ��ʾ��$�1�9�8�,�"��	��ھʾ����������A�M�Z�k�s�y�v�f�Z�M�A�4�(�����(�4�A�I�V�a�b�e�b�V�I�=�I�I�I�I�I�I�I�I�I�I�I�����!�-�0�8�8�-�!��������պں�����������������������������������������𼽼��ʼѼּؼּռʼü��������������������(�5�>�A�N�Z�f�Z�V�N�I�A�5�2�(�'�(�(�(�(�������������s�r�s�u�������������������������ɺѺֺ������Ӻɺƺ��������������	�������	� �������������	�	�	�	�	�B�N�[�g�^�[�N�E�B�9�B�B�B�B�B�B�B�B�B�BĦĳĿ����������������ĿĳĪĦĠěĚěĦĚĦĨīĭĩĦĚĘđĘĘĚĚĚĚĚĚĚĚ��(�1�4�6�7�4�+�(��������������`�l�y�������~�����x�l�`�S�Q�L�K�O�S�]�`D�D�D�D�D�D�D�D�D�D�D�D�D�D~DrDuD~D�D�D��ܻ�������������ܻлû����������л��{ÀÇÍÓàáçèàÓÇÂ�z�v�p�r�v�z�{�'�4�:�@�M�N�M�D�@�4�'�%�����'�'�'�'ǈǔǡǬǭǶǭǬǡǠǔǈ�~�{�u�x�{ǅǈǈE�E�E�E�E�E�E�E�E�E�E�E�E�E�E|EzE{E{E�E��I�G�K�M�Q�]�n�{ŇŋŗśśŘŒŇ�{�b�U�I = = =  { X   8  ( \ E * +  W 5 8 7 T J G F N [ # ) < 4 - I d > @ " Z [ > \ 3 D , , A v h V I R  r  r , * f 4 * S "  �  �  `  �  m  �  &  �  �  3  �  �  �  &  �  *  J    w  >  3  a  �  �  �  f  �  �  �  `  R  �  �  C  �  �  z    �  �  7  �    �  �  =  4  �  1  �  �    �  �  �  A  �    �  R  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  Bv  6  +  !        �  �  �  �  �  �  �  �  �  r  S  5     �  �  �  �  �  �  �  �  �  �  �  �  q  T  1  	  �  �  E  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  &   �  �    ?  Q  X  [  _  a  b  Y  G  )     �  �    �  q    g  o  d  Y  K  =  +       �  �  �  �  �  o  Z  E  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  5  �  t  �  �    '  (  #      �  �  �  �  �  _  "  �  �  �  �   �  �  ;  e  �  �  �  �  �  �  �  �  �  �  �  d  $  �  <  �  �  �  T  �  �  �          �  �  �  �  d    �    p  �  �  #            
      ,  .  $            �  �  �  b  S  E  7  '      �  �  �  �  �  g  C    �  �  �  �  �  L  G  A  :  1  &    
  �  �  �  �  �  �  �  �  �  |  �  �  h  j  i  k  x  �  �  �  �  �  �  �  x  l  _  R  A  #    �  	s  
y  B  �  y    m  �  �  �  �  /  �  �  7  �  /  	�  �  �  L  `  q    �  �  �  l  P  .    �  �  w  C    �  v  A    �  �  �  �  {  n  ^  N  ;  (    �  �  �  �  [    �     �           �  �  �  �  �  �  �  �  �  y  i  Y  H  3        Y  �  �  �  �  �  �  �  x  E    �  v    �  
  Q  i  \  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  W  @  (    �  �  �  �  �  �  �  �  �  �  �    ~  }  {  z  �  �  �  �  �            �  �  �  �  �  �  �  t  P  &  �  �  L  �  _  �  �  �  �  �  �  �  ~  w  n  d  Y  J  :  '    �  �  �  �  T  J  ?  2  $      �  �  �  �  �  �  l  X  O  L  I  D  F  0  ?  F  J  E  7  "    �  �  �  h  ?    �  �  �  l  G    \  V  P  J  D  >  8  2  ,  &          �  �  �  �  �  �  Q  	  	@  	u  	�  	�  	�  	�  	�  	U  	  �  x  �  .  �  �    �  �  �  �  �  �  �  ~  j  U  >  #    �  �  �  S    �  �  O    �  �  �  �  �  �  �  �  �  o  G    �  �  q  *  �  �  H  �  z  �  �    u  i  Z  J  5      �  �  �  k  /  �  �  d     n  l  k  j  j  k  k  j  j  h  f  d  `  ]  V  K  @  �  w    �  �  �  a  :    �  �  }  �  ^  C    �  �  <  �  W  �  �  �  �  �  �  x  ^  v  j  O  3    �  �  �  [    �  a  �  )  S  C  2      �  �  �  �  �  �  k  Q  3    �  �  �  d  +  	E  	�  
  
K  
b  
f  
[  
D  
$  	�  	�  	u  	  �  9  �  �  �  Z  B  �  �  �  �  �  �  �  �  h  B    �  �  Y    �  R  �  �  �  n  f  ]  P  C  4  %      �  �  �  �  �  q  M  '  �        :  2  %      �  �  �  �  Q  +  �  �  �  U     �  �  �  �  �  �  �  �  �  �  �  �  u  C  	  �  z  %  �  o  Z    �  �  �  �  �  b  5    �  �  �  P    �  �  5  �  \  �  �    �  �  �  �  �  �  �  �  �  u  _  G  -    �  �  7  �  �   �  �  �  �  �  �  �  y  p  g  _  T  H  =  1  %    
  �  �  �  �  �  �  s  _  H  -    �  �  �  �  �  �  l  9  �  �  T  �  (      �  �  �  �  �  �  �  q  [  I  7  #    �  �  �  �  (  5  ?  J  Q  V  W  N  6    �  �  �  h  :    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  n  e  ]  T  K  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  Z  B  *     �  �  �  �  �  �  �  �  u  m  c  T  D  4  #    �  �  �  y  R    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  o  ^  M  :  &    �  �  �  �  �  �  =  <  8  1  $    �  �  �  �  [    �  e  �  �  �    �  �  �  �  �  x  c  N  9  $      �  �  �  �  �  �  �  y  Z  �  �  �  �  �  v  a  J  .    �  �  �  �  �  a  G  4  (     "  �  �  �  �  �  �  �  k  J  %  �  �  �  I  �  �  4  �  ^  q  z  r  T  *  �  �  8  �  S  �  X  �    f  �  q  
�  �  �  �  �  �  �          �  �  f    �  
�  
/  	B  4  �  �  �  �  �  g  ?    �  �      �  �  X  �  v  �  }     �  Z   �      �  �  �  �  �  �  �  �  p  O  *  �  �  d  �  �  $  �  �  �  �  �  c  H  -    �  �  �  |  R  #  �  �  �  X    �  �  �  �  }  k  Q  ,  1  E  ,  �  �  n    �  >  �  ]  �  [  
�  
Q  
  	�  	�  	>  �  �  B  �  }  #  �  �  4  �  �    �   �
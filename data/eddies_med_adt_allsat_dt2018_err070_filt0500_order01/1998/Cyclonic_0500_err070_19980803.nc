CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?���S���       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�xd   max       P�,!       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �� �   max       <e`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E�Q��     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v��z�H     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�s�           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       ;�o       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�߿   max       B-V�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B-��       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >m�   max       C���       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >D�   max       C���       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          l       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          M       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�xd   max       P[��       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�i�B���   max       ?�҈�p:�       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��-   max       <T��       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E�Q��     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?������    max       @v���
=p     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P`           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�@            V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ER   max         ER       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�IQ���   max       ?�҈�p:�     P  X�               $                  ^                              l               	      +                                       	   	         6      J      D   
         !      "               (   
   #   -      Ot�:N�'O�RNIߥO�NO���N��[N��xO^�jOG53P�,!O?�/N�&?PR�9M�xdP!lGN�N�@�O�xN}�PK��O6�#N���N��hO�M�N��O���P�9N�vN	�N�.P�O
�8N��N�6OE��O��O��;N�O0�NA�N���O�_�O�XP/OPՎP7f�N��5P��N��aO��OZϕO��&N���O��@N��Nb�vN���P	�~O���N�>�O}��Oy�lO"��O,,<e`B<T��<49X<o;ě�;D��:�o�D���D�����
���
��`B�o�o�#�
�#�
�49X�T���T���e`B�e`B�e`B��1��j��h��h��h�����������o�C��\)�t���P��P����w�#�
�,1�0 Ž0 Ž49X�49X�49X�@��D���T���Y��Y��Y��Y��Y��]/��o�����7L��7L��\)����������� �����������������������

�������������������������������������������������%/6BO[hnoqpvtl[OB6'%;BDN[gktx|�~tg[N?:;��������������������1;FHT]aceaTKH;111111����������������������	"/1/,)%	���������� ����������qy����������������tqHHT[addcbaaVTTOLLIHHHnz��������zaH<163hg[Z[`hntvthhhhhhhhh����������������������������������}2<>HUY^__UH<92222222t}��������������pzxt��������������������
#0IVcjlsnbU0��<@FIU^bfebXUTI<868:<#/265//$#��������������������#*=DN[gn�����tgB5)!#�����

���������htv���������������thUan|���������zUPLKMUHOS[hktvtthc[ZOLHHHH+5BNPPNB;5++++++++++�����������������������
#IUbny��{b<0���@BKN[glpgda[YNB=;;@@�����������������������

�����������")-5BN[gfc][PNB5)'""�������������������������
���������FHJTUXYWTHEDBDFFFFFF��'06861�����stv������tssssssssss��������������������M[g����������zg[NLHMcgot{��������tge^_cc�����)8BHI<������������������������������&*&�������� #/2771/&#""$/Haz����znfaK</%""jnsz���������~ztnkjj������������������������������������������

�������������������������������������������������r{|�����������{zusrr)68BCEB6+)��������������������X]\r�������������taX����������������������������������������257BIN[`gtxmg[NIB822�
/<E<71/,# �]ahnz����������zqn_]')5@BNGC;5,)�
�������������������
��#�0�<�>�/�#��
�����������������ļʼʼӼʼ��������������4�1�(��$�(�4�A�M�Z�c�f�k�f�b�Z�M�A�4�4¦£¦¨²½¿¿¿¼²¦¦¦¦¦¦¦¦�T�;�5�0�,�,�4�;�G�T�`�m�y�����������m�T�Ŀ����������������Ŀѿ���������ݿѿ��U�a�n�r�v�n�a�Z�U�H�F�F�H�T�U�U�U�U�U�U���	���	���"�)�(�"� ��������$����������
��$�(�0�2�=�@�@�=�2�0�$�����������������������	�����	�����������������������	�/�H�z���������n�H�"��Y�M�J�C�@�;�@�M�N�Y�f�r����������r�[�Y�[�Y�J�O�R�[�h�tāčďĚġĚčā�t�h�a�[��������������������/�H�W�_�\�R�;�	�������ݿٿѿͿѿۿݿ޿��������꿟�y�X�<�1�2�;�G�h�v�������ĿϿ߿�ɿ����ݽӽн˽Ͻнڽݽ������������ݽݽݽ������������������������������������������������������������������������������������������������������������������������z�`�]�a�s�y�������н��� ����ݽĽ��2�(��� �(�4�A�M�U�f�q�s�|�{�s�f�M�A�2�H�A�<�7�9�<�H�U�a�b�n�o�p�n�a�U�H�H�H�H���������������������������������������޾���׾ʾþ������վ��	�������	������������������%�(�$����������������x�s�h�c�e�j�l�x���������������������9�Z�g�s�������������n�g�Z�N�5��ù������������ù̹Ϲ۹ܹ޹ܹӹϹùùùýн˽ŽϽнݽ��ݽҽннннннннно�������������������������������������������������������������������������������Ňń�{�w�w�u�{ŇŎŔŠūŭŷŹŭŠŔŇŇŹŵŭťŭŹ����������������ŹŹŹŹŹŹ�������������	������������I�?�=�:�7�3�7�=�I�V�b�f�o�o�q�o�d�b�V�I�������������Ŀѿݿ�������ݿѿĿ���������ŹųŴź����������������������Z�P�Z�c�g�s�������������s�g�Z�Z�Z�Z�Z�Z�s�q�g�f�e�k�s�������������������������sù÷ìæãìù����ùùùùùùùùùùù�G�?�=�G�T�U�`�m�w�m�k�a�`�T�G�G�G�G�G�G���x�o�m�j�a�a�m���������������������������������������������������������޼�	����#�%�#�'�2�@�M�Y�o�n�f�Y�@�'��t�r�f�r�~���������ɺֺ������ֺ������t�Y�c�w�����ּ����� �!������ʼ��r�YE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFF FFF<FVFcF}F�F�F�F�F�F|FhFVF=F1F�Ŀ����������������ĿοѿտѿͿѿӿѿĿ�ƎƁ�u�\�Q�_�h�uƁƚƪƼƽ����ƺƳƧƚƎ���ݿѿ����������������Ŀѿܿݿ�����������������
��$�=�I�T�Z�X�G�+�$������3�+�,�3�@�L�T�Y�e�k�r�z�r�l�e�Y�O�L�@�3�����������������ɺֺ��������ֺɺ�������� �����'�4�@�B�A�A�@�4�'��ùðìéëìøù����������úùùùùùù�5�4�(�'�'�(�5�A�N�Z�e�Z�T�N�A�9�5�5�5�5ĳĦĚā�u�Z�V�[�tĦĳĿ���������	����ĳ����ĿĻķķĿ����������������������ػl�b�_�T�_�l�x���������������������x�l�l��������������������������!� ��������E)EED�D�D�EEEEE*E7ECEPEXEWEPECE7E)�����������������������ʾ׾ؾݾ۾׾Ծʾ�ÓÇÂ�z�y�s�n�l�nÇÓàåìîîìæàÓ g ! 7 4 ' * O i ? + \ H r A : W @ $ 3 t & J 8 ` A . Q F * m = { H H 0 0 W ' r : J ] O  �  f 9 ^ N D W O U < + S H \ ' | ) 3 P X    C  �  .  Z  �    z  �  �  �  %  �  9  �  	  �  �  �  Z  t  \  �  �  �  �    �  �  �  x  4    6  �  �  �  2  �  �  �  x  �  �  H  �  B  �  �  0  �  H    �  1  o  �  �  �  �  �  '  �    �  t�o;D��;�o:�o������49X���
������C��ȴ9��`B����C��e`B�#�
��1����#�
��o���ٽC���w��h�L�ͽ�P�e`B����P�`�+�+�ixս0 Ž��#�
�y�#�u����8Q�]/�H�9�L�ͽq���P�`�ȴ9��o���7L��h�}󶽋C����
��{����� Žy�#���P����ě���"ѽ�����"ѽ��9X��Bl�B$�Bs�B>vB�B��B!��A���B5�A�߿BHB�cA��pBqqB�oB+l5B)�BH	Bq�B�fB%r�B&��B�B��B�B!�B &�BI�BR'B��B!3�B%��Bx�B�2B#��B�rB�B�:A��EBQ�B
eB�B	׬B	��BZCB!5JB-V�B�@B�iB�B;B)�B�lBg�B5�B)=�B}�Bx�B
��B�NBB|�Bq+B��BߌB@B$�BxB@qBM�B��B!��A��B��A���BԀB[�A�`�B��B��B*�0B)�bB�BEB�B%B8B&�bB��B@TB	vB xB��B��BD5BƳB!D�B&G;B?�B��B#B��B?�B!)A��+B@+B
�QB��B	@�B	�B@kB!;B-��B��B��B�B��BLBL9BBJB�B)@B��BMB
��BB�tB@|B@<B��B��A�'�@�i�A;� A�1�Aj�A{HZA�?�A���B	�A�uA�#@ܭ�A�"�A��A|�)An<^A,Q^A�A��xA�$�A#��A<��A�rA�
AW~_A�12@���A�) >m�A)�UAMU+A��A�&A�p@���B��Ayw�A�'HA��A�AA��Ag��A��}A��5@��@&>0@���C�c8C���Ax�B�@Az��B	�?˘�@/�@��AͲXA��#A��A�i@��A���C���ANe#A�7A�{!@�A<�A�Aj��Az2�AƇA�Y]B	¨A�dA��@��A�*nA�y�A|��Am9A,T�A���A�{LA�u�A#=~A;�:A�\�AЊtAW�Aԅ>@��VA��;>D�A*�AMjA���A�TA�z�@�E�B�TA{�wA�`�A���A�w�A�{�Ag�A�ßA���@�ai@3��A�2C�e�C���Axd�B@�Az�B	N�?���@6p1@ɲ�A�9A�srAހ]A�~@�-A�z4C���AOd?A�}�               %                  ^                              l               	      ,                                       
   	      	   7      J      E            !      "         	      )   
   #   -                     #                  M         3      3               3                     )            3                  #               !      +      9      /            #                  /                                 !                  5         %      1                                    #            3                                       +      1      +            #                  -                  O\\�N�8�N��NIߥO���O���N��[N��xO^�jO�+P[��N�Q�N�]�P �^M�xdP��N9�)N�@�N���N}�O�spO6�#N�.�N\e�O�M�N��O�E�O�i�NEv�N	�N�.P�O
�8N��N�6O0�O��O�V�N]��O"I�NA�N���Om"O�XP�ZO+3xP2ON�ËP�N}�3O��OZϕO��&N�O�LN��Nb�vN���P��O383N�>�Oq|aOo�O"��O$�  �  C  u  [  �  �  6  j  5     O  u  �  K    �  �  �  V    	[  [    �  �  e  �  �  L  )  �  l  �  5  �  x  �  �    �  Q  \  �  �  D  �  �  �  }  �  �  �  V    �  2  �  >  p    �  �  	   �  ;<T��<D��<#�
<o;o;D��:�o�D���D���o�+�e`B�t��e`B�#�
�49X�e`B�T����/�e`B�m�h�e`B��9X�ě���h��h���C��t��������o�C��\)����P��w��w�#�
�#�
�,1�<j�0 Ž8Q�<j�8Q�D���T���]/�Y��Y��Y��]/�]/�]/��o�����C�������\)���P���P�����-����������������������

��������������������������������������������������)3BO[hgnmspmg[OB6.*);BDN[gktx|�~tg[N?:;��������������������1;FHT]aceaTKH;111111�����������������������	"%&$"	�����������������������|������������������|HHRTXaddbaa_TPMLJHHH9FAHanz������zaUH<69hg[Z[`hntvthhhhhhhhh����������������������������������������2<>HUY^__UH<92222222����������������������������������������	#0<@JNNLE=0#		<@FIU^bfebXUTI<868:<#/154/.#��������������������#*=DN[gn�����tgB5)!#�����

���������z�����������������uzUanz���������zURNMOULOZ[^hqnh[ROLLLLLLLL+5BNPPNB;5++++++++++�����������������������
#IUbny��{b<0���@BKN[glpgda[YNB=;;@@�����������������������

�����������$)05BN[ada[NKB:5)(#$������������������������

��������EHTUWXVTHFDBEEEEEEEE�&)/4.)�����stv������tssssssssss��������������������LP[gt���������sg[NMLcgot{��������tge^_cc�����)6BGH;��������������������������������%*%��������!#/2660/##'/<Haz���znaM</+##xz����������znxxxxxx������������������������������������������

�������������������������������������������������r{|�����������{zusrr)68BCEB6+)��������������������Y^]r�������������tbY����������������������������������������358BKN[^gswmg[NJB933�
/:961/+# �]ahnz����������zqn_]')5BDIFB;5+)�
���������������������
��#�.�:�.�#��
�����������������üȼɼ������������������A�7�4�(�!�(�(�4�A�M�Z�a�d�[�Z�M�A�A�A�A¦£¦¨²½¿¿¿¼²¦¦¦¦¦¦¦¦�T�;�4�0�1�;�G�T�`�m�y�������������y�m�T�Ŀ����������������Ŀѿ���������ݿѿ��U�a�n�r�v�n�a�Z�U�H�F�F�H�T�U�U�U�U�U�U���	���	���"�)�(�"� ��������$����������
��$�(�0�2�=�@�@�=�2�0�$������������������� �	����	�������������������������������	�/�T�c�c�H�/��	�׼Y�T�M�I�H�M�Q�W�Y�f�r�����{�r�h�f�]�Y�[�[�O�M�O�T�[�h�tāĉčĘčā�t�h�[�[�[�"�	������������������0�H�M�Z�Z�V�K�;�"����ݿٿѿͿѿۿݿ޿��������꿟�y�Y�=�2�3�;�G�e�s�������ĿͿݿ޿ǿ������ݽнϽнؽݽ���������������������������������������������������������������������������� �������������������������������������������������������������z�{�������������Ľнݽ��ݽн��2�(��� �(�4�A�M�U�f�q�s�|�{�s�f�M�A�2�H�B�<�7�:�<�H�U�`�a�m�n�n�n�a�U�H�H�H�H���������������������������������������޾���׾ʾþ������վ��	�������	������������������%�(�$��������������x�t�j�e�g�l�x�������������������������-�<�Z�g�s�����������k�g�Z�N�5��ù����������ùϹֹعϹʹùùùùùùùýн˽ŽϽнݽ��ݽҽннннннннно�������������������������������������������������������������������������������Ňń�{�w�w�u�{ŇŎŔŠūŭŷŹŭŠŔŇŇŹŵŭťŭŹ����������������ŹŹŹŹŹŹ�������������	������������I�B�=�;�8�5�:�=�I�V�c�n�o�p�o�n�c�b�V�I�������������Ŀѿݿ�������ݿѿĿ�������ŹŴŶż������������
�������������g�[�f�g�s�������������s�g�g�g�g�g�g�g�g�s�g�g�f�g�l�s�x�����������������������sù÷ìæãìù����ùùùùùùùùùùù�G�?�=�G�T�U�`�m�w�m�k�a�`�T�G�G�G�G�G�G�����{�r�q�q�l�m�z�������������������������������������������������������޼�����&�#�'�4�@�M�Y�n�n�f�Y�@�4���~�z�~�����������ɺֺ����ֺ��������~�f�`�d�x�����׼���� �!������ʼ��r�fE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�FFFFFF!F>FVFcF|F�F�F�F�F|FgFTFJF1F���������������Ŀ˿ѿӿѿȿĿ�����������ƎƁ�u�\�Q�_�h�uƁƚƪƼƽ����ƺƳƧƚƎ���ݿѿ����������������Ŀѿܿݿ�����������������
��$�=�I�T�Z�X�G�+�$������3�,�-�3�6�@�L�W�Y�e�r�t�r�k�e�Y�N�L�@�3�����������������ɺֺ��������ֺɺ�������� �����'�4�@�B�A�A�@�4�'��ùðìéëìøù����������úùùùùùù�5�4�(�'�'�(�5�A�N�Z�e�Z�T�N�A�9�5�5�5�5ĳĦĚā�v�[�W�[�h�tĦĳ�������������ĳ��������ĿĽľ������������������������ػl�b�_�T�_�l�x���������������������x�l�l�������������������������	����������E*EEED�D�D�E
EEE*E7ECEPEWEVEPECE7E*�����������������������ʾ׾ؾݾ۾׾Ծʾ�ÓÇÂ�z�y�t�n�n�zÇÓàåìîíìåàÓ a  > 4 $ * O i ? & Y I k 4 : Y : $ 7 t - J 0 g A . D D ? m = { H H 0 / W   e 3 J ] T  � u h 8 Z ; D W O S 6 + S H \  | ' 4 P Q    �  �  �  Z  s    z  �  �    )  �    U  	  �  b  �  �  t  P  �  �  �  �    %  A  h  x  4    6  �  �  {  2  r  �  c  x  �    H  �  �  �  �  �  {  H    �    G  �  �  �  �  z  '  �  �  �  \  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  ER  �  �  �  �  �  �  �  g  N  =  /    �  �  �  h  5  �  `  �  +  8  C  ?  ;  5  .  $      �  �  �  �  �  �  `  9    �  c  i  p  t  t  s  p  k  f  Y  L  =  $    �  �  �  �  �  ~  [  K  ;  +      �  �  �  �  �  �  v  ^  F  .    �  �  �  B  p  �  �  �  u  h  \  R  W  Q  C  +    �  �    p  �  5  �  �  �  �  �  �  u  b  P  ?  ,    �  �  �  i  #  �  �  J  6  &  $  5  :  ;  :  3  )  '  "    
  �  �  �  �  �  �  �  j  d  _  Z  T  O  I  D  >  9  2  *  !      	     �   �   �  5  )        �  �  �  �  x  M  1    �  �  c  �  q  �  i  �  �  �        �  �  �  �  �  �  �  �  ~  [  /  �  �  R   �  �  Q  �    9  J  N  A  -    �  z  "  �  <  �  �  �  �  2  X  _  f  l  q  t  u  t  o  e  T  4    �  �  ^  h  H  
  ;  �  �  �  �  �  e  ;    �  �  �  �  �  �  �  r  ^  K  ?  7  �    !  8  J  G  @  <  1  %    �  �  �  ~  .    �  t   �    �  �  �  �  �  �  �  �  �  �  �  �  }  q  d  X  K  ?  2  �  �  �  x  d  O  H  N  `  h  Z  E  %  �  �  |  5    �  j  Y  f  s    �  �  �  �  �  }  p  a  Q  @  /      �  �  �  �  �  �  �  �  �  �  �    o  [  C  *    �  �  �  �  s  P  �  �  �  
  6  @  C  K  Q  U  U  D    �  �  �  O  "  *  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    u  k    �    }  �  		  	2  	R  	[  	N  	.  	  �      �  �  �  .    [  S  P  :  )      �  �  �  �  l  H  $  �  �  �  o  4              �  �  �  �  �  �  n  4  �  �  z  8  �  �    �  �  �  �  �  �  �  �  �  �  y  V  3  (  N  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  N    �  t  0  �  e  O  :  %    �  �  �  �  �  �  �  o  S  7    �  �  �  �  �  �  �  �  �  �  i  C    �  �  �  `  *  �  �  :  �  �  a  �  �  �  �  �  �  �  �  g  6  �  �  j    �  E  �    �  �  z  �  �    )  C  P  S  R  N  C  5     �  �  �  :  �  �  9  )  +  -  /  0  2  4  6  8  :  :  9  8  7  6  4  3  2  1  0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  g  ]  T  J  l  _  H    �  �  �  h  6  �  �  �  q  :    �  �  �  >   �  �  �  �  �  �  �  �  �  t  `  L  9  &    �  �  g    �  \  5  +  "        �  �  �  �  �  �  �  �    x  t  p  l  h  �  �  �  �  �  �  ~  v  p  n  l  k  h  d  `  ]  Y  V  S  P  q  w  v  m  X  ;    �  �  �  G    �  }  -  �  W  �  �   �  �  �  �  �  �  �  �  ~  `  ?      �  �  �  O     �  R  �  �  �  �  �  �  j  P  0    �  �  X    �  �  B  �  �  I  @  �  �  	      	    �  �  �  �  �  �  �  �  �  �  �  n  E  �  �  �  �  �  �  �  �  �  v  M  !  �  �  �  L  �  �  O   �  Q  D  7  *         �  �  �  �  �  �    g  O  5      �  \  G  1                �  �  �  �  �  �  s  \  F  .    �  �  �  �  �  �  �  �  �  �  �  �  �  a  @    �  �  �  8  �  �  �  �  �  �  �  �  �  �  y  o  e  X  K  ,    �  �  �  B  ?  $  �  �  �  ~  ]  M  -  �  �  h    �  m  �  '  Z    �  w  �  ~  �  {  g  V  O  K  :  !    �  �  �  �  �  +  t  �  �  �  �  �  U  .  �  �  U  u  �  Q    �  !  {  �  R  �  �  �  {  X  3  	  �  �  g  ,  �  �  b    �  �  -  �  X  �  I  }  v  S    
�  
�  
-  	�  	T  �  c  �  K  �  �  /  A  "  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  ]  I  5  �  �  �  �  �  �  k  P  4    �  �  �  �  ~  c  H  &  �  �  *   �  �  m  D    �  �  {  M  '     �  �  �  =  �  �  .  �  �  �  V  /    �  �  P    �  s  #  �  �  t  e  D    �  �     �  x  }  z  o  \  H  3      �  �  �  e  3  �  �  z  6  �  u  �  �  �  �  �  �  o  c  c  [  O  0    �  {    �    r  �  2  ,  &           �  �  �  �  �  �  �  m  L  +     �   �  �  z  p  ^  I  )    �  �  �  i  ?    �  �  �  n  D  ?  >  >  /      �  �  �  �  �  �  x  _  F  -    �  �    �     c  g  G  <    �  �  �  �  �  �  v  J  $  �  �  \  �  _  �  �                   �  �  �  Q  
  �  s  �  f  �  �  �  �  �  u  Z  =  !    �  �  �  �  z  Y  3  	  �  �  b  %  �  �  �  �  �  �  h  C    �  �  �  e  +  �  l  �  �    	  �  �  �  �  �  �  �  �  �  l  F    �    w  �  �  �  �  l  �  �  �  �  �  �  �  �  �    t  n  f  V  8  
  �  �  9  �    4    �  �  �  �  �  R    �  �  @  �  �  �  3  m  �  �
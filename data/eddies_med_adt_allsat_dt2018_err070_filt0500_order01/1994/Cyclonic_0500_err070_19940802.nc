CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�dZ�1       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�sm   max       P�W�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��;d   max       <�h       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @Fu\(�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\)    max       @v�          
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @O@           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�c            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �z�   max       <�j       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��w   max       B1�4       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�G�   max       B1U*       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?R"{   max       C���       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?@   max       C��m       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          l       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�sm   max       P�W�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�����+   max       ?��7��3�       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��;d   max       <�h       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @Fnz�G�     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @v��z�H     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @O@           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?�   max         ?�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?r��n/   max       ?���Fs��     �  ]                        %   !               k               	                  S            
   4      )               8               	   1                  	      #         &         \   "   J            9      (   @      0      N�Nz��N���PG[N}�O/J}OϑPW�'P'OiO���N���O�l5Nq�$P�W�O�0�N��N3*N;NY�N�x�NP�N�gN���O�}�P�q�O�YMO���O�4N���O�ߠN"�zP��O�WOD9�O�N�|NO▌OMʢNʮ�Oܵ�N�N� �O�N�u�N�?)OzP7OTkN^N�΍O	�9O�N(N��N�0�O�*�O6wEO#�P��P�
O���OB��O���O��,P�gM�smO��O�@�N�.�O�٫Ob�N��0<�h<e`B;D���o�o��o��o�t��#�
�49X�D���T���u�u���㼛�㼛�㼬1��1��9X��j�ě��ě����ͼ��ͼ��ͼ���������/��/��`B��h���������������C��t��#�
�#�
�',1�,1�49X�49X�49X�49X�8Q�D���L�ͽT���T���T���Y��aG��e`B�e`B�ixսu�y�#�}󶽅���7L��7L��^5��^5��;d������������������������������������������������������������-5Ng����������gB74��������������������+/6<>HU[__]UH@<1/$&+Wanz��������zunha\WW����'',)��������37BN[����������g[B53��������������������y{|������������{{yy��������������������&)58BCB=5)#i��������
���qai126;HLOW]eijbT;742/1����������������������������������������')6BEEB6+)''''''''''�������������������������������� �������������7;@TamnmcaTQJH?;:827��������������������N[f]UV[t������tg^NIN#0A^n{����ymI0
��{�������������{wust{�������"'�������#0<CCBC?0#���������������%*6B[hswwurlb[OFB52%457ABCB=51..44444444#IUbr���nbI<3)$
���������������������������������������������������������������
��������)+7CO\hu~����u\OC6+)��������������������aghrtw������tgebaaaa����#&",5/#
�������������������������������������������������
#7;9;6+
������/02<IJJJIID?<0/.-.//������������)5BN[figb^[NB65.)#���������������������������������������������������	����������������������������lnsx�������������zol26BOQVUOB8602222222236BFOQV[OB@650333333���������������������
#'/554//&#���`ajnz�������zsnaa^^`<TaduyunaUH</#������������������������
#/HSQH</)
���x{�������������{vqtx�������������r�������������trrpor��)696.#	�����������������������������xommuz����������7<HUalnuzz|snaUHC967�������������z{~�����+BNYPKGB9)����[[aghot��������tg[[[FHUaalnrnfaaULHFFFFF������������������������
�����
�����U�Q�Q�T�U�Y�a�f�n�o�q�n�c�a�U�U�U�U�U�U��
����!�-�/�:�@�C�:�-�!�������z�m�X�e�h�c�[�Q�T�m�z�����������������z�������(�,�5�8�@�5�0�(�������g�c�Z�P�N�F�C�N�Z�g�s���������������s�g������þþÿ�����������������������������a�\�]�l�r�r�������������������������s�a�����������	��5�A�E�A�5�6�4�6����;�0�+�+�/�?�C�H�T�Y�a�b�_�b�u�m�a�T�H�;���������������Ľнݽ����ݽнĽ����������s�M�A�4�#�4�A�h����������ѾҾʾ����B�>�5�,�2�5�>�B�H�N�R�R�N�D�B�B�B�B�B�B�I�$��������������$�=�V�oǎǔǅ�{�b�I���������������������	��"�!���	���������{�s�g�e�g�g�s�s�u��������������������ììàÞÜÚàçìðùùùììììììì�m�j�c�i�m�y�}�}�y�v�m�m�m�m�m�m�m�m�m�m�	�	���	���"�%�/�/�/�"��	�	�	�	�	�	�ֻܻ׻ݻ������������������ܾ������ľʾ׾޾ؾ׾ʾ��������������������(�&�(�0�0�2�5�7�A�C�I�N�Z�]�Z�V�N�A�5�(�����������������������������������������R�L�;�.��	��������	���=�H�X�`�d�`�R���i�Z�R�N�A�7�5�N�g���������������������ٿſ������Ŀѿݿ������)�5�(�����ٿ`�G�;�.��&�G�T�c�g�m�y����������y�m�`���ܻл������������лܻ����"�������g�d�Z�Y�X�Z�f�g�s����������u�s�g�g�g�g�����}�}�}�������������������	�����������)�+�6�>�B�H�B�6�)��������������!�-�F�_�e�c�g�y��x�l�S�0�!��<�0�#��	���#�<�I�U�b�n���v�d�U�I�<������ĿĶĶĴĿ�������������������
������
���������������������#� �����������	���#�$�0�0�1�0�$�$�����������׾ʾ��������������׾�������	��������ŵŭųŹ������������������������ƳƧƧƚƙƚƤƧƳ��������������ƳƳƳƳ����ûõöþ�������)�6�<�@�4�3�*�����M�L�M�M�Y�f�j�g�f�Y�M�M�M�M�M�M�M�M�M�M�r�f�e�Y�U�Y�e�h�r�~���������������~�r�r�Ľ����������������Ľнݽ������ݽĽ����߽��������(�/�0�(�������ŠŝŠŤŭŹ��������ŹŭŠŠŠŠŠŠŠŠ�������������������#�)�&���������ŭŠŔŋŇ�~ŇŔŠŹ����������������Źŭ�*�%� ����*�6�C�E�C�B�6�0�*�*�*�*�*�*ŠŔŔŇņŇŔŔŠũŭŹŽźŹŵűŭŠŠ��������������� �����������ܻл����������ûлܻ����
������U�I�H�N�U�a�n�r�s�n�k�a�U�U�U�U�U�U�U�U�_�]�]�_�g�l�x���������y�x�l�_�_�_�_�_�_�t�g�S�B�6�8�B�N�[�g�t¦¨¢����������������������)�,�3�4�)��Y�U�L�J�H�G�@�L�Y�e�k�r�x�~�����~�r�e�YE�E�E�E�E�E�E�FF$FGFbFiFeFZFIF=F$FFE��
������ĿĶ���������
�'�?�L�T�\�T�<�#�
E*EEED�EEEEEEE+E/E6ECEPETEOECE*�'�#���
��'�4�=�M�Y�`�\�Y�X�Q�M�@�4�'��{������������!�(�$��	�����ʼ��ā�w�x�{āčĚĦĶ����������ĻĳĦĚčā�ʼ����q�^�Z�f������ʼ޼��&�.�!��ּ�ā��t�h�g�h�tāćčďčāāāāāāāā�����������������z�v�n�p�t�{�����������������������������ʾ���"�*�5�+����׾������������������ʾ־׾���׾ʾ���������Ç�n�b�Y�U�L�F�H�aÇÓáìüÿúñàÓÇ�<�<�/�$�#�����#�%�/�<�H�G�E�E�=�<�<�����������������ÿĿĿĿ��������������� _ n ( f H > T G b ^ f w P @ } # s N Y Q ^ _ ; d S P T B / J S L Q L ? n . \ 7 U : 6 ; g V . 9 @ V D A , L 3 N W ; O Y 4 z = { v H T S J = H  	  �  �    �  y  g  �  c  �  �  B  �  �  �  �  �  E  x  >  .  3  �  �  �  d  �  �  �  �  [    X  �  H  �  �  �  �  ;  3  �  �  �  �    �  {  �  ;  ,  �  �  ~  �  e  �  �  |  �    4  m  7  �  6  �  M  6  �<�j�o��o�����`B��C��e`B�<j�0 ż��ͼ�9X�,1������#�o��h���ͼ�/��h�����ͼ������Y���/�D���@��T����P���w�C���\)�<j�,1�@��\)��{�'0 Žy�#�0 ŽH�9��E��P�`�@��m�h�e`B�<j�T���q�����
��C������Q콓t���hs�zὶE������㽰 Ž�����������
=�����P�O߽�S����#B��B^JB,k�B	�BB�kB�B[�B��B
!�BH�B)U�B ��B��B�tA��wB�eB
fB� B!��Bp�B�A��B;�B	�	B&;�B)��BxB%8�B��B�RBEaB&��Bh�B��B�BpB1�4B�tB	�7B�B! B 2(B$	_B&4�B[hB�OB�mB`
B��B ��B �BH�BI�B��BT�BG{BS�BIB�B)�rB-��B
� B&SB'�B�]B�:B�Bu�B	�gB��B��B:B,@�B	��B��B07B~�B��B
ERB�B)%�B ��B �B=!A��B�B�sB�mB!k{B��B��A�G�B�B
>}B&=hB)��BձB%@1BD�B�PBA�B&��B=�B?�B��B@"B1U*B>vB	��B�~B!1\B ?�B#�(B&B�Bx�B �B�oBg�B��B!<�BLB@�BCuB@.BH�B��B��B?�B��B)�_B,�5B9GB@B4�BBB��B")BM�B
�B�A�-�A��@p�yA���A�JtA���A�W�A���A��A��/A(,�AC6A���B A��kA��6A���Ak�fA�Ŏ@��AP�A��A��}A]gA�3�A��Ahy�@�[A��A���A��@{ǳA�F�A䈹B�3B	2UATtxA�6	Bo�AҴ/@�b�?���A%��A1�A�B�A��gA��A��A��??R"{@���A�~�@���A�c�A���?�uAC���A�8>C��&@�3@�: A���@��[A�A��AU�IAO�A�N�A�Au`�A�}$Aƀ@m��A�qmA���A��A��A��A���A�_�A'HAAAKA���B?�A�}VA���A��Ak	A��%@�~3AP��A�VyA��A\��A�oAժAh�x@�>A���A��A�3}@w�A삘A�;B�~B	�AR��A���B��A�9i@�F@
�A%	iA1��A�(�A�hRA�FMB <�A�<�?@@���A�l@�9OA�~�A��e?�aaC��mA�rC���@��A�Aޢ:@�$�A�x{A�� AR�AN�'AɛbA�i?Ats�                        &   "               l               	                  T               4      *               8               
   2                  	      $         '         ]   #   K            :      )   @      0                  )            -   /         '      9   !                           %   ;      %   #      )      )               #         '         !                                          )   )   !      -      1         %      %                              -   +               9   !                              9      %   #      %      !                        '                                                   '   '         -      1                     N�NJ��N�IOw��N}�N��EN��OPW�'PO�6N���Ot�N/�)P�W�O�0�N���N3*N;NY�N�-�NP�N�gN���O:a�P���OS*xO���O��2N���O�G(N"�zO��O�WO-,O�N�|NO���OMʢNʮ�Oܵ�N�N� �OcC�N�u�N�?)Oc��O)��N^N�΍N�4�O4��NKN�*aO�*�O��O#�P�UO��O��OB��O���O��,P�gM�smO��O��N5OO��"Ob�N��0  V  �  �  9  �  /  �    �  �  �  �  �  
y  z  �  -  r  f  �    �    �    \  �  �  �    �  �  �    d  �  �  �  �  M  �  �  �  �  e  )  F  �  j    3  �  �  !  m  9    m  �  �    1  	  '  �  	�  P  4  o  ;<�h<D��:�o�o�o��`B��`B�t��D���D���D����j��o�u���㼣�
���㼬1��1��j��j�ě��ě��+������������/��/����`B�#�
���������,1�����C��t��#�
�]/�',1�0 Ž<j�49X�49X�@��Y��]/�P�`�T���aG��T���ixսq�����w�e`B�ixսu�y�#�}󶽅����网hs������^5��;d������������������������������������������������������������FYgt�����������g[QHF��������������������-/09<DHUW\]XUNH</,--fnz�������zpneffffff����'',)��������5BN[����������g[NB55��������������������y{|������������{{yy��������������������)355BBB:5)%i��������
���qai126;HLOW]eijbT;742/1����������������������������������������')6BEEB6+)''''''''''��������������������������������� �������������7;@TamnmcaTQJH?;:827��������������������egt���������tgg_[]ee�#0Dn{����xlI<0���w{��������������{xvw�������"'�������#0<B@A=0#
���������������+6B[hotuspj`[OKB62*+457ABCB=51..44444444#0IUbiy|{ubUI<0,+&#���������������������������������������������������������������
��������3:CO\hu~���uhC620/3��������������������aghrtw������tgebaaaa����#&",5/#
�������������������������������������������������
$((#
�������/02<IJJJIID?<0/.-.//������������%)5BNY[efa\[NB85/*)%���������������������������������������������������	����������������������������wz}������������zursw469BKORPOB?64444444446BEOQVXOBA661444444���������������������� 	
#/231/,#
��`ajnz�������zsnaa^^`<HUarvsmaUH</#����������������������
#%##
�������x{�������������{vqtx�������������r�������������trrpor��)696.#	�����������������������������xommuz����������<AHUaglmpqqmeaUHB><<�����������������������5BEJHFB<)���[[aghot��������tg[[[FHUaalnrnfaaULHFFFFF������������������������
�����
�����U�S�R�U�V�Z�a�e�n�n�o�n�a�]�U�U�U�U�U�U�����!�-�:�;�=�:�-�!�����������m�l�m�n�m�j�a�m�z���������������������������(�,�5�8�@�5�0�(�������s�j�g�Z�V�N�K�I�N�Z�g�s�y�����������s�s�����������������������������������������a�\�]�l�r�r�������������������������s�a������������5�A�D�=�3�4�2�5�1����;�3�.�.�1�;�@�B�F�H�T�`�]�a�t�m�a�T�H�;���������������Ľнݽ����ݽнĽ������M�K�F�M�W�Z�f�s�����������������s�Z�M�B�6�5�5�5�@�B�E�N�P�P�N�B�B�B�B�B�B�B�B�I�$��������������$�=�V�oǎǔǅ�{�b�I���������������������	��"�!���	���������|�s�g�f�g�j�s������������������������ììàÞÜÚàçìðùùùììììììì�m�j�c�i�m�y�}�}�y�v�m�m�m�m�m�m�m�m�m�m�	�	���	���"�%�/�/�/�"��	�	�	�	�	�	�ܻػػܻ޻���������
�������ܻܾ������ľʾ׾޾ؾ׾ʾ��������������������(�&�(�0�0�2�5�7�A�C�I�N�Z�]�Z�V�N�A�5�(�����������������������������������������������������	��"�.�0�9�:�/�.�"��	���������i�Z�S�N�;�C�Z�g�������������������׿��ѿ˿¿¿ɿѿݿ����	���������`�G�;�.��&�G�T�c�g�m�y����������y�m�`���ܻл������������л����!��������g�d�Z�Y�X�Z�f�g�s����������u�s�g�g�g�g�������������������������	��������������)�+�6�>�B�H�B�6�)��������������!�-�:�H�S�[�_�l�n�_�S�F�:�!��<�0�#��	���#�<�I�U�b�n���v�d�U�I�<����������ĿķķĶĿ����������������������
���������������������#� �����������	���#�$�0�0�1�0�$�$�����������ʾ������������ʾ׾�����������׾�����ŵŭųŹ������������������������ƳƧƧƚƙƚƤƧƳ��������������ƳƳƳƳ����ûõöþ�������)�6�<�@�4�3�*�����M�L�M�M�Y�f�j�g�f�Y�M�M�M�M�M�M�M�M�M�M�r�f�e�Y�U�Y�e�h�r�~���������������~�r�r���������������������нݽ��߽ݽҽнĽ������߽��������(�/�0�(�������ŠŝŠŤŭŹ��������ŹŭŠŠŠŠŠŠŠŠ�������������������� �(�$���������ŸŭŠŔŐňŔřŠŭŹ��������������ŻŸ�*�%� ����*�6�C�E�C�B�6�0�*�*�*�*�*�*ŠŔŔŇņŇŔŔŠũŭŹŽźŹŵűŭŠŠ�����������������������ܻջл»��ûĻлܻ���������������a�[�U�O�U�X�a�n�o�p�n�c�a�a�a�a�a�a�a�a�_�]�]�_�h�l�x���������x�x�l�_�_�_�_�_�_�t�g�S�B�6�8�B�N�[�g�t¦¨¢�)�'�������������������)�0�0�*�)�Y�U�L�J�H�G�@�L�Y�e�k�r�x�~�����~�r�e�YE�E�E�E�E�E�E�FF$FDF_FgFdFYFGF=F$FFE���
���������������
�#�:�G�P�T�W�K�<�#�EEE
EEEE'E*E7E7ECEKEOEEECE7E*EEE�'�#���
��'�4�=�M�Y�`�\�Y�X�Q�M�@�4�'��{������������!�(�$��	�����ʼ��ā�w�x�{āčĚĦĶ����������ĻĳĦĚčā�ʼ����q�^�Z�f������ʼ޼��&�.�!��ּ�ā��t�h�g�h�tāćčďčāāāāāāāā�����������������z�v�n�p�t�{�������������������������ʾ׾�����"�&�� ��׾ʾ������������ʾ˾оʾ���������������������ÓÇ�z�n�f�]�V�T�U�a�zÇÓàôøòäàÓ�<�<�/�$�#�����#�%�/�<�H�G�E�E�=�<�<�����������������ÿĿĿĿ��������������� _ n G _ H > L G e [ f P S @ } % s N Y G ^ _ ;  R < T @ / F S D Q 6 ? n + \ 7 U : 6 ) g V ) 6 @ V E - 1 K 3 F W 6 O # 4 z = { v H a F ; = H  	  �  @  &  �    �  �    W  �  k  i  �  �  �  �  E  x    .  3  �  �  �  �  �  �  �  ,  [  l  X  {  H  �  q  �  �  ;  3  �  �  �  �  �  j  {  �  �  y  :  �  ~  Z  e  �  7  )  �    4  m  7  �  \    �  6  �  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  ?�  V  A  +       �  �  �  �  �  |  f  P  A  A  @  >  7  /  '  �  �  �  �  �  �  �  �  �  �  '  �  �  :  �  �  	  	d  	�  	�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     +  #      )  7  3  !    	  �  �  �  �  l  :  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  H      $  *  .  /  ,  &      �  �  �  �  �  �  �  s  W    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  ]  E  /            �  �  �  �  �  �  �  �  F  R    �  l  �  }  _  l  �  �  �  �  �  f  $  �  �  �  �  v  B  3  %  �  �  �  I  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  n  \  J  ;  ,    �  �  �  �  �  x  m  a  T  E  3       �  �  �  �  |  B  �  �  a  Q  1  e  m  |  �  �  �  �  ^  (  �  �  S  �  �    N   e  �  �  �  �  �  �  �  �  �  �  w  o  g  ^  V  R  O  L  I  F  
y  
d  
@  
  	�  	V  	5  	  �  	�  	�  	1  �  �  t  �  �  �  �  �  z  l  \  H  3      �  �  �  �  {  R  $  �  �  q  (   �   �  �  �  �  �  �  u  [  =    �  �  �  �  b  -  �  �  �  [  #  -      �  �  �  �  �  �  �  s  :    �  �  �  ^  6     �  r  l  f  `  V  J  ?  '  	  �  �  �  �  �  k  P  5     �   �  f  T  B  2  "    �  �  �  �  �  �  �  q  \  F  /  "  !  !  �  �  �  �  �  �  �  l  L  )    �  �  y  4    �  �  K          �  �  �  �  �  �  �  �  �  �  y  _  F  -     �   �  �  |  r  h  ^  T  J  @  6  ,  "         �   �   �   �   �   �    �  �  �  �  �  �  �  �  �  ~  t  ^  G  +    �  �  �  K      B  �  �  �  �  �  �  �  j  H    �  �  5  �  �  �   �  �  	  �  �  �  �  �  �  n  E    �  �  {  $  �  '  |  �  l  %  E  S  Y  [  W  K  ;  '    �  �  �  I    �  �  7  �  H  �  �  �  �  �  r  c  c  `  T  @  $    �  �  �  �  S    �  �  �  z  w  y  m  X  @  &      .  V  L  -    �  �  \    �  �  �  �  �  �  g  C    �  �  �  �  i  C      �  �  �  �      �  �  �  �  �  �  q  Q  !  �  |    �  �  j  �    �  �  �    }  z  w  u  s  q  m  j  f  b  _  [  Y  ]  `  d  W  �  �  �  �  �  �  �  i  @    �  �  y  ;  �  n  %  �    �  �  �  i  ?    �  �  �  `  @  &    �  �    "  �  �  �  �      �  �  �  �  �  �  �  �  w  c  I     �  �  �  �  �  d  ^  T  H  <  -      �  �  �  �  R    �  �  #  �     �  �  �  �  �  �  �  �  �  �  �  m  Q  -    �  �  �  �  �  �  6  ]  ~  �  �  �  �  �  s  M    �  �  a    �    t  �  3  �  �  �  �  �  �  u  [  <    �  �  �  y  �  �  �  �  U  )  �  �  �  �  �  u  \  >    �  �  �  f  9    �  �  �  p  V  M  "  �  �  �  �  o  R  4    �  �  �  �  [    �  �  M    �  �  �  �  �  �  �  �  �  ~  j  V  =  %    �  �  W  
  �  �  �  �  �  �  �  �  q  [  F  0      �  �  �  �  �  i    8  {  �  �  �  �  �  �  �  �  �  �  ^  5  �  �  =  �  b  [  �  �  �  �  w  e  R  ;  "    �  �  �  j  9    �  �  d  *  e  _  Z  T  N  H  C  =  7  2  -  (  !      	    �  �  �    &  "      �  �  �  �  �  �  e  >    �  �  n  L  :    ?  B  E  F  A  9  1  $      �  �  �  �  X  *  �  �  �  J  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  Q  8  !  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �  �  �  w  ^  J  <  2  &    �  0  �  �    !  1  ,      �  �  �  {  A     �  x  ,  �  ]  �  �    F  m  �  �  �  �  �  ]    �  G  �  `  �  a  �  N   �  �  �  �  �  �  �  �  |  V  -     �  �  m  @    �  �  �  m  !      �  �  �  i  7     �  �  d  d  Y  >  �  W  w  w   a  l  m  l  m  k  f  [  J  0    �  �  |  >  �  �  X    �  �  9  3  +     
  �    �  �  �  �  W    �  �  .  �  �    V  �    �  �  �  �  �  W    �  j  �  ;  
a  	m  ^  *  �  �   �  E  e  l  j  e  _  Y  S  Q  E  '     �  �  R  �  �  �  P   �  	�  
>  �  '  _  y  �  v  j  T  5    �  q  
�  	�  �  �  �  g  �  �  �  p  d  H  &    �  �  y  ?  �  �  o  "  �  F   �   ,    �  �  �  �  �  [  *  �  �  �  \  I    �  y  :  	  ~   �  1    �  �  �  �  }  p  \  =    �  �  |  K     4  S  q  �  	  	  �  �  �  	  	  �  �  r    �  E  �  U  �     �      '    
  �  �  �  �  �  �  �  �  �  |  g  C    �  �  �  �  �  �  {  a  ?    �  �  y  >  �  �  b  �  �    �    �  p  �  �  	%  	S  	x  	�  	v  	M  	  �  �  �  T    �  &  ;    �    �  �  �      %  2  ;  D  I  M  Q  T  X  X  S  N     �  �  �    "  2  *    �  �  �  b    �  �  (  �  ;  �  �  H  �  o  ]  A  $    �  �  �  j  8  �  �  v  -  �  �  G  �  f  �  ;    �  �  `  %  �  �  r  4  �  �  �  J    �  �  o  7  �
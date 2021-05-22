CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�KƧ     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       Pë�     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       <�o     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F��
=p�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �޸Q�    max       @vQG�z�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�H          $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���#   max       ;��
     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5
�     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��~   max       B4��     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =0�   max       C�K�     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =t �   max       C�Hs     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          \     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          E     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P�T�     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Y��|��   max       ?ҔFs���     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       <�o     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?0��
=q   max       @F��
=p�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��z�G�    max       @vN�G�{     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�~        max       @�G�         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?ҔFs���     �  _�      Q   L      !   
   !      &   8      2      =      
         +   	            [      8   (                                 /         '      &         "      '         
      
   /      
         !   &                  2   	               
   Ne2�Pë�P�~pNk�O��zN��O��O� �O�-�PC�GO	�P��JOu�=Pn�~O��N���N��NN�OԯiN,EN���NȁN4P"G�O�P&I O��wO���N��TN�IeN+�fO^1N���NQ�lNO��N��O��O��O��OA�YP5GO�H�OŰwNx&O@�9O�؎O�s|O��oN��RNaO��Nto�O/�O4�/O��N�a�O6M Ou�O�[�P��NX��N�oO��N���O#��PXFN��O��vN�$�N�Z�N��N0njN��<�o;D��;D��;o���
�o�t��49X�D���e`B�e`B�e`B�e`B�u��o��C���t���t���t����㼛�㼴9X��9X��j��j��`B��`B��h��h�o�o�+�+�\)�t��t��t����#�
�#�
�0 ŽD���H�9�H�9�L�ͽP�`�P�`�T���aG��aG��aG��aG��e`B�ixսm�h�q���}󶽃o�����+��+��7L��7L��C���C���t���t���t����
��E���E���������#+/07;/#���0Y{�����{Z<0���y����� ���
������wy8<IUbbb^UTI<;5888888�����
##
�����������������������������"/<CF@9/
������
.<HhnpVZUH?/�
��������������������:Taz������zcTH?8016::<FHUadmnonleaUPHG=:s~�������������tkqps��������������������iz�������������zmeeiqt��������������thjqFHRU[acejaUH><?AFFFF/<?ED<6/-///////////������������������������)40-(����������������������������������������������QUUanrznaUQQQQQQQQQQ?BOP[][UOGBA????????5BN[gt�xlg[B>;511536CS\hju~��uh\VOF963em������������kc^[]e��������������������3;HTmz�������maT?5/3BBEOU[ghojh[OBBBBBBB�����������������������������������)646BFOPOJLJGB6)('%)56>BDORUQONB?7665555��������������������������������@BNUUPONNJCB@@@@@@@@W[ght��������tlg\[WW����)0)��������������������������������������������������������������egmqrt~���������tlje���
#,/<GD<2
������JU[anxwnkaWUJJJJJJJJ��tpgfcdggmty�������Z[gt�����������tpbZ`ntz����������tg_\\`�����������������������������������������������������������GNSUagnrxz}znfaUHBG��������������������ggt��������|tligfddg���

	 ������6HUanpx|}zwnaULH<306��������������������FHU_aknqwwvonaUHF??F��������������������5B^gnuv��}[LC5���#'$!�������t{�������{wttttttttt									��������������������;BN[giig[NB;;;;;;;;;#<?<;<?<50#*BOgt����������g[/)*���������������������������������������������zqnqzz{��������"#&046780,#-0<@EIIKKI><:10-++,- 
         JNR[gt{|vtgf[ZTNJJJJ�z�t�i�g�d�g�t�	�������g�F�5�!�"�Z����������������	���y�`�E�6�3�G�`�����Ŀѿݿ������������������������ʼռּܼؼּʼ������������������������Ż���������$�.�1�'����������������������������ʾ˾ʾʾþ�����ìàßÓÇÈÑàìù����������������ùì�h�a�T�H�E�9�/�$�/�a�����������������z�h��������������$�'�0�7�=�B�A�=�9�0�$�����������������-�3�7�H�Z�^�T�H��	��������������������������������	���������ؾ����s�i�m�s���������"�����	�㾾���H�A�D�C�M�Y�f�r���������������r�f�Y�H�h�H�4�2�7�/�8�3�*�C�\�uƳ������ƻƚƁ�h�H�B�<�B�B�;�?�H�a�n�ÇÀ�z�m�u�v�n�a�H�����������������������������������ĳıĳĿ��������Ŀĳĳĳĳĳĳĳĳĳĳĳ���
���
���#�#�#� ���������s�g�b�^�d�j�m�l�s���������������������s����������	�������������������������Z�N�N�E�B�C�N�Z�\�g�q�s�v�s�o�g�Z�Z�Z�Z�U�I�H�G�G�H�U�Y�Y�X�U�U�U�U�U�U�U�U�U�U�a�^�a�a�m�n�q�zÄ�z�s�n�a�a�a�a�a�a�a�a�ڿοʿ̿Կ����5�N�g�~���g�A�(�����ڿ���	�����"�.�;�A�@�>�;�6�.�"��ĳĨĞĢĜĠĿ������#�7�0�'�
��������ĳ�^�>�I�L�Z�n�{ŔŠŭŹ������ŹŭŐ�{�n�^��	������������(�5�9�H�J�E�5�(���Ϲ͹ù����������ùϹҹڹڹйϹϹϹϹϹϿy�n�m�`�^�T�T�G�T�`�l�m�y�~���������y�y�����"�.�3�8�.�"���������������������������	��"�/�7�2�2�/�"���������������������������������������������������������������������������������������������������������������������������ʾ��¾ʾ׾�����׾վʾʾʾʾʾʾʾʾ������������	�������	������M�:�����$�@�F�R�f�r����������{�r�M�$�����)�6�B�O�[�e�h�t�j�h�[�O�B�6�$àÛÓÓÎÉÇÌÓàçìù������üùìà�r�g�d�j�t��������!�*�-�!�����㼱���r�$��������������������$�0�>�F�G�=�0�$�ѿ˿������������Ŀѿۿ���������ݿѿĿ����������ĿϿѿտѿοĿĿĿĿĿĿĿ��\�g�j�h�a�\�O�C�6�5�*� ����*�6�C�O�\���������������������	��"�*��	��������a�H�@�;�3�0�;�C�H�T�a�s�����������z�m�a�t�m�h�e�k�uāčĚĦĺľļĵĳĦĚčā�t���������������������������	���������������������$�)�$���������Z�N�A�>�;�A�G�N�Z�a�g�s�z�x�s�p�m�o�g�Z��������������������������������������������������)�.�5�>�B�M�B�5�3�)���D�D�D�D�D�D�D�EEEE*E,E*EEEE	ED�D칄�����������Ϲܹ������Ϲɹ�������������~�{���������������������������������ݾ׾Ծ׾پ����	���� ���	�����<�0�#����#�0�<�I�P�Y�f�p�n�g�b�U�I�<ŭ�z�i�U�J�I�<�I�b�n�{ŕŭ������������ŭ�-�!���
�!�.�G�y���������������l�`�G�-�����'�4�@�D�@�;�4�'����������������������������������'�3�@�E�L�Y�\�_�e�h�b�Y�L�@�3�'��Ľ��������ĽнѽսؽڽнĽĽĽĽĽĽĽĻ������������л׻ܻ߻�����ܻлû�������¹�{�w¦¿�������5�C�;��������r�g�j�r�~�����������������~�r�r�r�r�r�r��ܺܺ׺����!�-�F�W�P�F�:�-������⾙������������������������������������������������������Ľн����ݽٽнĽ���������
���(�5�A�C�A�4�2�(������h�b�e�h�t�~āĉćā�z�t�h�h�h�h�h�h�h�hĦĤĦĩħĬĳĿ����������ĿĳĨĦĦĦĦ I d > ] R & N ` 4 6 (  4 J j < G E / Z > t L V s / � M G c Y J : Y � y = Y - N u E M ^ , Q 9 . B i G g 7 D d J ! J � @ B i h V I P h _ V F \ ^ k    �  i  �  �  �  �  �  �  O  I  '    �    �  �  .  j  �  9  �  r  D  (  �  �    �  �     D  ;  �  [  �  `  O  G    �  �  $  �  �  �  �  .  )    �  ;  �  t  �  �  �  �  	  x  �  q  P  �  �  q  �  �  W  :      j  �;��
���w����o��㼋C��'��H�9��O߼�/��o��㽙���8Q���������ͽy�#��`B���ͼ�/��/��xռ�`B��1��C��e`B�8Q��w�\)�#�
�#�
��w�,1�#�
�@���{��o��%���T��%�� Že`B���w��1�����^5��+�u��o�}󶽅��������T��C���j��1�Ƨ������hs��\)��1��t�����#���T�Ƨ������`��
=��/��`BB�!B&6�B+:�B&��B0(B5
�B8B��B��A��B��B �B )B �vB �B�<B��BЅB�pB
B�&BNqB�bB�CB1�B �)BE�A��B"xB�mB�B�BV�B��B%hB�B	�{BB�B'+B�yB-[B
V�B��B?�B
#�B
"�B
3BΩB��BzBD1B�RB
	lBLZB�BٸBNB|�B(?BΝB(�BW�B�!BYB%bUB	��B�B܆B�B%e=B&5B��B	?�B�kB'pB*̓B&��B<8B4��B��BGB��A��~B��B �'B 7GB �B ��B��B��B�TB��B>�B�>B�tB��B�GB24B �B@A�B=nB��B��BğB=6B��B��B?�B
)�B@B;�BךB--aB
? B��B�B
2*B	�0B
�
B�B�B@�Bf�B-oB	��BCB�"B��B�4B�NB��B?]B)=DB?�B��B@�B%>�B	�5B�[B�KB8)B%@B&?�B��B	@A��A�N�Ar�@���A�H~AL�KA�/�A���B	mEA��A�aSAP%@�BU�A��lA��BA�#A�E\A�_A�`bA���A�u�A�n�A���A_	A��A�U�A�"�>ui�Aj�A_0FA��A��A��A��eAS7rAY�`@�@�A���A���@��rB	3A}��AxquB �1A���A���A��}A��)B	bA��"A��A��C�K�=0�@��MAY@�A��wA�,HA<�@��A/d�?���A'��@��;A�7@�x@f��AH��A&%�A4��A܆2A��cA�w'A�eYAo�@���A�v�AL�8A�t�A�x�B	~;A��=A��AN�@�Bj�AŀA�ðA�A�iA�xWA��7A��Aœ�AǌJA�^3A`�!A�A�}�A�{\>M�]Ajp7A_7gA��GA��zA�|iA��*AQ�AZC@��A؀�A�|�AE�B��A};�Ax�B ��A�;7A�~XA��A�ZB�`A�v�A�k�A��C�Hs=t �@��pAZ�UA�}�A��A�t@���A.�D?�85A',H@���A���@z�@p�%AH��A%	A7 .A܍AႤ      R   M      "   
   !      &   8      3      =      
         ,   
            \      9   )                                 0         '      &         "      (         
      
   0      
         !   '                  3   
               
         I   ?      !      #   )      -      3      7   !                           -      )   %                                 +         3      !         !                           !            /   )                  9                           E   5               )            -                                                                              #         3      !                                                +   )                  9                     Ne2�P�T�Pa��N2��On��N��Oj��O�t�Od2OTN�C�PB��Ou�=Oǰ�O�W^N�֮N��NN�O��NN,ENMk�NȁN4O�LFO�O�%�N��VO���NE��N��zN+�fO^1N���NQ�lNO��N��N�4~O���Ob?xOA�YP5GO�H�O�	DNE��O@�9Os[KO�s|OY��N��RNaO��Nto�O/�O
��O�N�a�O ~O6�sO��P��NX��N�oN�k�N���O#��PXFN��Oq�0N�$�N��aN��FN0njNl�5  �  I  �  �  X    �  �     �  B  i    �  �  <  P  *  �  �  �  �  x  W  �  9  �  �    9  �  �  �  �  ;  �  ,  p  �      z  �  E    2  $  �  C  X  O  z    W  �  %  �  �  r  e  *      /  �  �  ]  �  �  k  �  !  �<�o�D���#�
:�o�#�
�o��C��D����t��49X�u�ě��e`B��㼛�㼓t���t���t���`B���㼬1��9X��9X�T����j�H�9�,1��h�+�+�o�+�+�\)�t��t����49X�,1�#�
�0 ŽD���Y��L�ͽL�ͽe`B�P�`�ixսaG��aG��aG��aG��e`B��o�q���q����o��7L��+��+��+��7L��t���C���C���t���t����P���
��j��^5�������#+/07;/#���	#<n{�����{U<0�������������������~��7<?IUZ[UPI<<77777777������

����������������������������
#/<?<71/#
����
/<HafcXUYRH/
	�
��������������������EHHTamtz|zsmfaTIHCAE<<GHUaclmjcaURIH><<<|���������������|}x|��������������������tz������������zuqnptt����������������umtHHLUZabdeaVUH>@BHHHH/<?ED<6/-///////////������������������������ ""�����������������������������������������������QUUanrznaUQQQQQQQQQQ?BOP[][UOGBA????????<BN[gvz{ztog[NLE=;:<36CS\hju~��uh\VOF963krz�����������zpkjjk��������������������3;HTmz�������maT?5/3HO[[\hlhg[OFHHHHHHHH�����������������������������������)646BFOPOJLJGB6)('%)56>BDORUQONB?7665555��������������������������������@BNUUPONNJCB@@@@@@@@_gqt��������tqga____����$*'�������������������������������������������������������������egmqrt~���������tlje���#'/<>?-
�������NU_anwvnhaZUNNNNNNNN��tpgfcdggmty�������egt���������tlge`]^e`ntz����������tg_\\`������������������������������������������������������������GNSUagnrxz}znfaUHBG��������������������ggt��������|tligfddg�����

�������1<HUacjov{znaUMH@<41��������������������GHQUanntutnmaUHG@@GG��������������������'5B_gt��|m[KB5���#'$!�������t{�������{wttttttttt									��������������������;BN[giig[NB;;;;;;;;;#<?<;<?<50#*BOgt����������g[/)*���������������������������������������������zqnqzz{��������#023420#/0<<<BHIJJIH=<20.--/ 
         U[gtzztqgg[ZUUUUUUUU�z�t�i�g�d�g�t���������g�T�D�0�-�5�T���������������	�����`�D�>�=�G�T�����Ŀѿֿٿݿ����𿸿����ʼ����������ʼҼּڼּּʼʼʼʼʼʼʼ�������������������������!�'�+�,���������������������������ʾ˾ʾʾþ�����ìêæàÖÔÖàìñù��������������ùì�j�a�T�I�G�:�4�;�H�T�a���������������z�j�����������$�0�6�;�=�>�=�=�5�0�$����������������	���"�'�/�3�8�/�-�"����������������������������������������ؾ����q�v�|�����������������׾����H�A�D�C�M�Y�f�r���������������r�f�Y�H�h�_�T�L�M�\�h�uƁƚƬƶƸƶƮƧƚƁ�u�h�G�@�E�D�=�A�H�Q�a�n�}Ã�y�n�k�m�r�n�a�G�����������������������������������ĳıĳĿ��������Ŀĳĳĳĳĳĳĳĳĳĳĳ���
���
���#�#�#� ���������s�j�c�f�k�r�s�������������������������s����������	�������������������������Z�W�N�L�J�N�Z�c�g�r�i�g�Z�Z�Z�Z�Z�Z�Z�Z�U�I�H�G�G�H�U�Y�Y�X�U�U�U�U�U�U�U�U�U�U�a�^�a�a�m�n�q�zÄ�z�s�n�a�a�a�a�a�a�a�a���ڿܿ������8�A�M�Q�G�A�(���������	�����"�.�;�A�@�>�;�6�.�"����ĿĸĲİİĵĿ���������������������b�V�X�b�e�n�{ŇŔŠŕŔŇŇ�{�n�b�b�b�b��	������������(�5�9�H�J�E�5�(���ù����������ù͹Ϲ׹ֹϹùùùùùùùÿy�s�m�`�_�V�T�Q�T�`�i�m�y�}�������~�y�y�����"�.�3�8�.�"���������������������������	��"�/�7�2�2�/�"���������������������������������������������������������������������������������������������������������������������������ʾ��¾ʾ׾�����׾վʾʾʾʾʾʾʾʾ�����������	�������	���������@�4������'�4�@�N�f�r������r�f�M�@�)��"�)�*�6�B�O�[�a�h�l�r�h�g�[�O�B�6�)àÛÓÓÎÉÇÌÓàçìù������üùìà�r�g�d�j�t��������!�*�-�!�����㼱���r�$��������������������$�0�>�F�G�=�0�$�οĿ��������ÿ¿Ŀѿ���	�������ݿοĿ����������ĿοѿԿѿʿĿĿĿĿĿĿĿ��\�g�j�h�a�\�O�C�6�5�*� ����*�6�C�O�\���������������������	������������������a�H�@�;�3�0�;�C�H�T�a�s�����������z�m�a�t�h�m�t�yāčĚĦīĳĶĹĶĳĦĚčā�t���������������������������	���������������������$�)�$���������Z�N�A�>�;�A�G�N�Z�a�g�s�z�x�s�p�m�o�g�Z��������������������������������������������������)�.�5�>�B�M�B�5�3�)���D�D�D�D�D�D�D�D�D�EEEEEEEEED�D������������������Ϲܹ����Ϲȹ�������������~�{����������������������������������߾׾�����	�������	������0�-�#����#�0�<�I�M�U�b�b�f�b�U�I�<�0ŭŔ�z�i�U�L�I�b�n�{ŇŔŭ������������ŭ�-�!���
�!�.�G�y���������������l�`�G�-�����'�4�@�D�@�;�4�'�������������������������������'�!�"�'�3�7�@�L�Y�[�Y�U�L�C�@�3�'�'�'�'�Ľ��������ĽнѽսؽڽнĽĽĽĽĽĽĽĻ������������л׻ܻ߻�����ܻлû�������¹�{�w¦¿�������5�C�;��������r�g�j�r�~�����������������~�r�r�r�r�r�r���ܺ�����!�-�F�U�O�F�:�-�!�����ﾙ��������������������������������������������������ĽнսܽнĽ���������������������������(�1�4�A�A�4�-�(����h�b�e�h�t�~āĉćā�z�t�h�h�h�h�h�h�h�hĳīĨĭĳĿ����������Ŀĳĳĳĳĳĳĳĳ I ^ 9 L Q & U O & ! ,  4  n : G E , Z C t L B s  O M D \ Y J : Y � y 7 W ( N u E J [ , B 9 ) B i G g 7 8 c J  > � @ B i Q V I P h ^ V ? Q ^ >    �  �  �  c  �  �    )  �  �    5  �  �  2  �  .  j    9  b  r  D  �  �  o  $  �  e  �  D  ;  �  [  �  `    �  �  �  �  $  �  }  �  �  .  �    �  ;  �  t  A  ?  �  N  �    �  q  P  �  �  q  �  �  �  :  �  �  j  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     
        =  E  8      �  �  H  �  �  �  :  �  h  �  �  8  �  �   �  `  �  �  �  �  �  �  �  �  �  W    �  p    �  �    \  s  �  �  �  �  �  �  �  v  j  _  N  7     	  �  �  �  �    a  +  4  E  W  P  C  A  I  L  C  0    �  �  a    �  �  O                  �  �  �  �  �  �  �  �  e  5     �   �  �    <  d  }  �  �  ~  t  _  N  '    �  �  �  �  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  "  �  �  �           
  �  �  �  �  �  R    �  v    �    g  e  �  �  �  �  �  �  �  �  �  �  �  �  �  �  R  �  g    �  �  A  B  @  :  )    �  �  �  �    Y  1    �  �  R    �  �  T  ^  b  h  h  d  [  I  -    �  �  d    �  r  N    �  �    �  �  �  �  �  �  �  ~  j  S  ;  #    �  �  �  9  �    �  �  �     �  �  �  �  �  �  |  i  M     �  Z  �  �  �   �  �  �  �  �  �  �  l  >    �  �  y  [  R  $  �  l  �  F  �  %  0  ;  ;  :  /  "        �  �  �  �  �  �  �  �  f  F  P  )  �  �  �  o  ]  .  �  �  U    �  i    �  p    �  m  *    �  �  �  �  �  �  y  b  J  3    	  �  �  �         �  �  �  �  �  �  �  �  y  a  A    �  �  [    �    �  �  �  �  �  �  �  �  �  �  �  ]    �  �  _    �  �  F   �   �  �  �  �  �  �  �  �  �  �  �  �  �  z  i  Y  I  8  $     �  �  �  �  �  	      )  .  )  %  !        �  �  �  �  �  x  s  o  j  c  U  G  9  )      �  �  o  !  �  �  �  \  7  
-  
�  
�    1  M  W  L  *  
�  
�  
�  
  	�  	  i  �  z  �  �  �  �  �  �  �  �  �  t  b  K  5       �   �   �   �   �   }   ]  W  �  �  �    #  2  8  ,    �  r    �  l  �  x  �  p   �  :  1    (  '    j  v  6  �  �  {  A  �  �  �  .  S  �    �  �  r  ]  F  .    �  �  �  l  /  �  �  z    �    h   �  �  �  �          �  �  �  �  �  �  �  v  W  +  �  }    8  8  9  6  +  !    �  �  �  �  �  s  Q  4      �  �  h  �  �  �  �  �  �  �  �  �  �  �  �  ~  w  l  b  W  L  B  7  �  �  �  �  �  �  �  �  �  �  ^  /  �  q  *     �   �   �   �  �  �  �  �  �  �  �  |  h  R  <  &    �  �  �  �  �  ]  8  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  p  C    �  ;  .  "    
  �  �  �  �  �  �  �  �  g  J  .    �  V  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  )  (  )  +  ,  +  %      �  �  �  �  �  m  I    �  �  H  I  I  L  X  2  '    �  �  �  Z     �  �  v    p  �    H  �  �  �  �  �  �  �  �  ~  ^  5    �  �  5  �  v  �  �   �      	  �  �  �  �  �  f  ;  	  �  �  6  �  u    �  �  B    �  �  �  �  �  �  ]  .  �  �  r  %  �  �  �  o  �  /  �  z  h  V  B  ,    �  �  �  �  �  �  �  �  �  �  �  �  c  A  �  �  �  �  �  �  �  k  H     �  �  �  v  @    �  8  �  ]  .  6  ?  E  F  H  F  B  >  ;  8  4  *      �  �  A  �   �      	    �  �  �  �  �  g  1  �  �  G  �  v     r  �  "  �  �  �  0     	  �  �  �  Z  $  �  �  w    �  3  �  �  �  $          �  �  �  �  �  �  x  S    �  �  9  �      �  �  �  �  �  �  �  �    R    �  �  [  
  �  �  �  �   �  C  5  %    �  �  �  �  �  �  �  a  @    �  �  �  N    �  X  S  N  J  C  3  $      �  �  �  �  �  ^  4  	  �  �  �  O  9  #    �  �  �  �  �  �  �  �  |  i  T  @  3  &      z  n  a  T  F  7  '      �  �  �  �  �  �  �  i  J  *      �  �  �  �  �  �  �  �  �  �  �  n  S  5    �  �  �  �    /  @  U  E  +    
�  
�  
H  	�  	x  �  v  �  �    t  �  �  �  �  �  �  �  �  T     �  �  }  A  �  �  q  $  �  P  �  ;  %     �  �  �  �  l  O  0    �  �  �  �  R  &    �  �  �  b  {  }  v  m  a  N  8      �  �  �  B  �  �  x  P    s  h  �  �  �  �  �  q  _  H  ,    �  �  w  A  �  �  
  Y   �  *  q  f  T  ;      �  �  {  9  �  �  p    �  &  :  �  �  e  I  &    �  �  _    F  B  &    �  �  v  9    �  ?  �  *  $            �  �  �  �  �  �  �  �  q  Z  B  *          �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  q  d  �  �  �          	       �  �  �  �  �  O    �  j  %  /  &      	  �  �  �  �  �  �  �  �  �  �  �  _  6     �  �  �  �  �  �  �  �  |  t  j  _  R  <  &    �  �  �  �  �  �  �  �  �  q  M  !  �  �  �  n  "  �  S  �  �  \  �  M  �  ]  P  C  7  *          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  p  e  P  4    �  �  u  1  �  �  �  U    �  c  @    �  �  �  x  G    �  �  V    �  Q  �  �  N  �  c  a  d  j  k  _  K  8  #  	  �  �  �  c  ,  �  �  f     �  �  �  �  �  �  �  �  �    L    �  �  ]    �  �  .  �  Z  !      �  �  �  �  �  �  �  �  j  O  &  �  �    �  6  �  b  �  �  �  z  R  (  �  �  �  w  I    �  �  �  }  T  +  
CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?š���o       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��*       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���m   max       <ě�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @FW
=p��     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��=p��
    max       @v|(�\     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @P            �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��            5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �L��   max       <�1       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�t   max       B0��       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B0@        9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >N`�   max       C��       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >?�/   max       C�Ƒ       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          Q       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       P��       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�Q   max       ?�d��7��       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �      max       <ě�       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @FH�\)     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @v|(�\     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P            �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E�   max         E�       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�|����?   max       ?�a|�Q     �  Y|               
   +   9            4      q               (         �                        '         	      P            ?      :                           B                  4      	          )   (               NW��ON;��N�A�NO��P�N���N�rN��PbҞN(5�P��*N���OW��NW�uOi#TO�{!N��O\��PNL�O��&OdŃN��,N�(�O��NE�O�5�O�4'O�O�oFN�$�N�!PBS�O+}�N#�N��P:��N���P�zO�aN��O Nm~9O&a�N�_%P"N�?�PY]�N��]OV�?O`�N߳N���O�[�Ne?�O�MOM2&O�W�O8{�O�iN���N�yN4S`N���N��j<ě�;�o:�o�o�D����o�49X�D���u��o��o��C����
��j���ͼ�`B��h�����o�o�+�+�\)�\)��P��P����������w��w�#�
�,1�,1�,1�49X�49X�8Q�<j�<j�H�9�H�9�T���T���ixսixսm�h�u�u�y�#�}󶽁%��%��o��7L��C���\)������������������������mFNQU[dgngd[NFFFFFFFFVadnz�������zvnga\VVUamoroma`XUUUUUUUUUU��������������������16BOPVOB:611111111116BOQWZ\XPB6)
�����������������������������������5;AHTWYUTJH;:4555555��������������������q|����������������sq��������������������������������������������������������������������������������������������st}�������������tsrs*6CP_krtsh\C*lt�����������ztpkkll�������  ���������� 5Ng���������g5)#��)6BMQB6&*���������
��������������	������)+67BFOSVOOB64*)"!!#/<>AA@>></#"�����������������������������������������������	��������������������)5>[t������tgNB51$��������������������DO[ahrt����|vth[OMDD������%&�����������������������������MOZ[]hhlh[OMMMMMMMMMehqtu{thbeeeeeeeeeet}������
!�����qt//49<HU^aUPHC</)////������
#/;/
�������������������������������������������������#$$%#
�������#/173/)#RTVamuz������vmaTTRR��������������������P[g~���������tg[NHJP9<>HIU_bnnqnfbZJI<<9S[jt���������tgdN<;Sv�����������|tvvvvvv��������������������gmz������������zlhdgX[]hotx�������th[[XX//14;<HMOJHE<///////<HNUYz�����znaUH<;:<��������������������{}~������������{zww{��������������������/4<HUanvvnlaUH<1/-,/����
#(#
������.5BN[g�����tg[VLB93.#&0<=@BFED<80+#!;<>DIIJI><;779;;;;;;���������������������$$����������������������������ʾs�j�f�Z�T�Z�]�f�o�s�}�{�s�s�s�s�s�s�s�s�û������������ûлܻ�����ܻллû����������������������������������������������!�-�3�A�:�:�-�!�������a�\�X�`�a�n�w�r�n�e�a�a�a�a�a�a�a�a�a�a�;�"������	��/�H�T�a�k�q�s�r�m�a�T�;��r�f�^�Y�X�Y�^�h�r�������������������H�@�<�4�/�,�/�<�H�U�a�k�n�o�n�m�a�U�H�H�t�m�h�d�d�h�tāĂčďĎčā�t�t�t�t�t�t�
�	����
��#�#�,�*�#��
�
�
�
�
�
�
�
�y�m�T�M�G�G�T�a�m�����Ŀ�������Ŀ��y�A�?�9�?�A�B�M�Z�]�Z�Y�M�A�A�A�A�A�A�A�A���`�S�K�Z����S�l�x���������h�^�F���������ݿۿӿѿ˿ѿݿ�������������������x�l�j�n�x�����������лԻлû�����ŭũŠŔŎőŔŠŭűŹŴŭŭŭŭŭŭŭŭ�s�o�f�e�j�s�������������������������s�	�����������	��"�.�3�6�7�5�.�%��	�����������������������ĽʽĽ���������������|����������������������������������$�������0�@�=�B�V�{ǃ�{�k�a�V�0�$�#�������&�'�"�(�4�E�K�M�Z�[�A�;�.�#�M�F�@�1�%�)�4�@�M�Y�f�r�u�|�~�y�r�f�Y�M�����������u�m�p�s�{�����������������������������������������������������������ā�t�h�[�O�B�6�3�3�4�6�B�O�[�h�t�w�yāā�����������������������������������������I�<�0�#���#�<�I�b�{ŔřŔőŇ�{�n�b�IŠśŝŭŴŹ����������	��������ŹŢŠ�׾ξʾɾʾ׾׾����	�	���	�����׾׾������������	��"�(�(�'�����	��ìääàßàìù����������ùìììììì����������¿¼º¿���������������������ؼf�^�f�n�����������!�������ּ���f�������������������������)�)������������������ûʻϻĻû����������������������������������������������������������s�p�f�`�b�h�����������������������������~�s�g�d�`�_�g�s�y�����������������������t�p�{�����˿׿ҿݿ����������ѿ������r�h�e�]�[�e�e�q�r�~���������������~�r�r�r�o�e�d�e�r�~�����~�r�r�r�r�r�r�r�r�r�r�z�x�z�}ÇÓÝàìùý����ûùìàÓÇ�z�H�D�G�H�U�a�h�n�u�n�a�U�H�H�H�H�H�H�H�H��������ƽ�����������������������������������������������������
�������������I�3�(���!�<�I�b�{ŇŔŞŞřřŗŇ�b�I���������������~�������������������������t�t�j�w¦¿������#�.������������������� ������������������������������������)�*�2�2�.�*�"����H�<�7�8�:�@�B�H�M�T�Z�`�b�e�t�y�m�a�T�H�ù������������������ùϹչҹѹչйϹù�FFFE�E�E�E�E�E�FFFF FFFFFFF�����������ĹιԹܹ�����������߹ù�����FFFFF$F1F8F4F1F$FFFFFFFFFF��������������'�4�@�B�H�A�@�4�'���ĿĽĺĽ�����������������������������̾�׾̾þ����žʾ׾��	����	������ED�D�EEEEE*E7E>ECELEMEIE<E7E*EEE�[�T�W�^�f�l�tāčĚĨīķĮĦĚďā�t�[�����������������Ľнݽ�ݽӽн˽Ľ������������	����(�(�(��������������������)�)�*�6�6�B�6�)���������)�$�!�'�)�6�B�F�C�B�8�6�)�)�)�)�)�)�)�)�.�"�.�3�:�G�S�`�e�`�^�S�G�:�.�.�.�.�.�. P 1 9 V h ) 8 0 1 = C > r X H : 2 P & G = W # b j G � m r ; k : u P l Q F [ _ y @ C > ? 7 ] 4 n 7 > ) Y \ T g + H L 1 o Y D p J a Z  �  4  _    d  !  r    �  �    I  
(  C  �  q  �  �      �  W  �  H    Y  �  �  �  5  �  �  5  �  �  :  '  �  �  �  D  )  c  y  s  4  w    �  �  �    1  �    o  D  �    �  �  	  s  d  �  �<�1�#�
�ě��D���T���@���7L����`B��j��C���1�+�\)�8Q�+�D����\)�'Y��L�ͽ0 Žm�h�''P�`�'u����Y���%�D���8Q���H�9�D���H�9����T����񪽁%�T���u�u��+��o�� Žy�#���m��+��Q콝�-���������������������xվO߽��S���xս��C�B��B)A��$B+��B7UBnwBcB!G�A�tB�AB+�B��B��B ��B!$OBK0BshB0��B��B��B&zB��B#D�B�BګB-�B�B�B/\B@�B	�B,�B��B--�B��B?GBBA;BPB+�B <6B"��B��B�A��%B�JB	ƄB'B�B	��B
�nB�eB ��B:�B�B�B�B)Y|B��Bq�BnYB��B%�B&>�B�4B��B�RB�CB=�A��3B+�]B@ BFhB=B!?<A��B��B*��B�B��B �B!8�B��BBB0@ B�vB@B �BC�B#@ B4SB�cB>�B�CB�(B><B@WB	v^B��B�B-BNBL�B~�B<+B?�B�B?�B� B"�]BB��A��B�ZB	SNB&�7B
2GB
}�B ?A��,B��B<NB��B:8B)?�B�'B@B�4B<>B%��B&?�BKQB%B�)AA��@�LOA�0�@i�A��A�]@�HA�B�A��A��PAr�A<n@L�dA��@�a�A��HAGh@A\�sA"�A�|�B
�A7��@ةiA�O~A���A�5�A���A��HA���AV-wAZ�+A��A�Ӏ@���A�w7@�~@���A�0^A��1Av�1@ hd?��A�Z�A���B�KB�AA���A���A��A�ZUA���A��j>kC���>N`�C��@���A�*AU��C���A�H�A%�?A2�yA־qA�+aA�UAA@��aA�x�@kKAƀ_A���@�4A�m�Aܤ�A�k-An�A<A@S��A�p�@���A�~�AF@�A[ �A!�A�|lB!�A75@���A��A�BnA�tA���A��A���AU4�A[$eA�m�A���A�~A�n�@��e@�A���A���Av�@�?�+PA�`�AŌ�B��B�3AA��A��A��pA�{[A�>?�/C���>�twC�Ƒ@�"NA䐐AW4C��DAܝ�A#A4�*A֌�A׀A�                  ,   9            5      q               )         �                        '         
      Q            ?   	   ;                           B                  5      
          *   (         	                        #   #            7      Q                        1   %                     !               1            3      /                     %      3                  #                                                                  3      C                           %                                    /                  )                           )                                                   NW��N�SyN;��N�<)NOt�CO��N�4N�rN��P\��N(5�P��N���OW��NW�uOX��Ol8�N��OP#�O<ZO��&O$j�N��,N�(�O��NE�O�5�O:�N��kO1U�N�$�N�!P3wO+}�N#�N��O��DN���O�3[O�aN��O Nm~9O&a�N�_%O�:N�?�P �"N��]O.�!O`�N�ԲN�O*�1Ne?�O�MN�w�OY/O8{�O�iN���N�yN4S`N���N��j  a  �  �  �  G  �  �  �  �  &    �  	�  �  �    �  �  �  �  �  7  �  1    �  �    d      �  j  �  �  �  �  e  �  	  �  �  �  �  
  �  �  �  j    �  m  �  �  	E    �  �  6  �  	�  �  �    	  i<ě�;o:�o��o�D�����
��C���o�u��o��C���C��\)��j���ͼ�`B����P���+�   �+���\)�\)��P��P���8Q�0 Ž8Q��w��w�49X�,1�,1�,1��O߽49X�T���<j�<j�H�9�H�9�T���T����+�ixս�O߽u��o�y�#������������o��7L���w���P������������������������mFNQU[dgngd[NFFFFFFFFhnz�������zynjcbhhhhUamoroma`XUUUUUUUUUU��������������������16BOPVOB:61111111111)6BGOPOKB6)������
���������������������������5;AHTWYUTJH;:4555555��������������������s~����������������ts�������������������������������������������������������������������������������������������������t��������������ytsrt*6CGOXcih\OC6*lt�����������ztpkkll������� ������������05=BNY[ekkgd[NB54.-0��)6BMQB6&*����������	
��������������	������)+67BFOSVOOB64*)"!!#/<>AA@>></#"��������������������������������������������� ��������������
���������@BN[gnt�����tg[NMB;@��������������������DO[ahrt����|vth[OMDD�������$���������������������������MOZ[]hhlh[OMMMMMMMMMehqtu{thbeeeeeeeeee��������������������//49<HU^aUPHC</)////�������
���������������������������������������������������#$$%#
�������#/173/)#RTVamuz������vmaTTRR��������������������V[gt��������tgb[RQTV9<>HIU_bnnqnfbZJI<<9OZgt����������t[LFDOv�����������|tvvvvvv��������������������gmz������������zlhdgY[`ht~������th_[YYYY7<?HLMIH<<;777777777GHU[eowz|zpaUKHEA@AG��������������������{}~������������{zww{��������������������.06<HUahnjha\UH<5/..����
#(#
������.5BN[g�����tg[VLB93.#&0<=@BFED<80+#!;<>DIIJI><;779;;;;;;���������������������$$����������������������������ʾs�j�f�Z�T�Z�]�f�o�s�}�{�s�s�s�s�s�s�s�s�����������ûлܻ�����ܻлû�������������������������������������������������!�-�.�:�<�:�5�-�!������a�\�X�`�a�n�w�r�n�e�a�a�a�a�a�a�a�a�a�a�;�/�"�����"�/�;�H�T�`�f�h�h�a�T�H�;�r�f�\�Z�\�a�k�r���������������������r�H�C�<�7�9�<�H�U�a�g�j�f�a�U�H�H�H�H�H�H�t�m�h�d�d�h�tāĂčďĎčā�t�t�t�t�t�t�
�	����
��#�#�,�*�#��
�
�
�
�
�
�
�
�y�m�T�N�H�H�T�c�m�����Ŀҿ�������Ŀ��y�A�?�9�?�A�B�M�Z�]�Z�Y�M�A�A�A�A�A�A�A�A�����g�\�j������-�S�l�x���`�X�Q�F�!�������ݿۿӿѿ˿ѿݿ�������������������x�l�j�n�x�����������лԻлû�����ŭũŠŔŎőŔŠŭűŹŴŭŭŭŭŭŭŭŭ�r�f�f�h�k�s�������������������������r������������	��"�-�0�1�1�0�*�"��	�������������������������ĽʽĽ��������������}�������������������������������������0�)�$�!� �$�%�0�=�I�O�V�\�`�_�X�V�I�=�0�#�������&�'�"�(�4�E�K�M�Z�[�A�;�.�#�Y�M�@�6�4�+�4�@�M�Y�f�j�r�v�w�t�r�f�b�Y�����������u�m�p�s�{�����������������������������������������������������������ā�t�h�[�O�B�6�3�3�4�6�B�O�[�h�t�w�yāā�����������������������������������������I�<�0�#���#�<�I�b�{ŔřŔőŇ�{�n�b�IŹűŹź��������������������������Ź��پ׾Ͼ׾���������������������������������	���!�!�����	��ìääàßàìù����������ùìììììì����������¿¼º¿���������������������ؼ���r�k�q��������������
����ּ��������������������������)�)������������������ûʻϻĻû��������������������������������������������������������s�p�k�m�s�����������������������������s���~�s�g�d�`�_�g�s�y�����������������������}�w���������Կݿ��������ѿĿ��������r�h�e�]�[�e�e�q�r�~���������������~�r�r�r�o�e�d�e�r�~�����~�r�r�r�r�r�r�r�r�r�r�z�x�z�}ÇÓÝàìùý����ûùìàÓÇ�z�H�D�G�H�U�a�h�n�u�n�a�U�H�H�H�H�H�H�H�H��������ƽ�����������������������������������������������������
�������������I�@�3�*�.�5�I�U�b�n�tŀŇŊŇŀ�n�b�U�I���������������~������������������������²�{¦�����������������²������������� �����������������������������������&�*�0�/�,�*�����H�<�7�8�:�@�B�H�M�T�Z�`�b�e�t�y�m�a�T�H�ù��������������ùϹѹϹϹӹϹ˹ùùù�E�E�E�E�E�E�FFFFFFE�E�E�E�E�E�E�E������������ùϹٹܹ�����ܹ׹Ϲù�����FFFFF$F1F8F4F1F$FFFFFFFFFF��������������'�4�@�B�H�A�@�4�'����������������������������������������ؾ��׾Ͼƾ¾ƾʾ׾���	����	� ����ED�D�EEEEE*E7E>ECELEMEIE<E7E*EEE�[�T�W�^�f�l�tāčĚĨīķĮĦĚďā�t�[�����������������Ľнݽ�ݽӽн˽Ľ������������	����(�(�(��������������������)�)�*�6�6�B�6�)���������)�$�!�'�)�6�B�F�C�B�8�6�)�)�)�)�)�)�)�)�.�"�.�3�:�G�S�`�e�`�^�S�G�:�.�.�.�.�.�. P ( 9 6 h  2   1 = < > b X H : 0 J & ?  W  b j G � m I @ Z : u I l Q F * _ n @ C > ? 7 ]  n . > ) Y e Y P + H - ( o Y D p J a Z  �  �  _  �  d  �    �  �  �  �  I  .  C  �  q  �  �    �  �  W  ]  H    Y  �  �  �  �  �  �  5  w  �  :  '    �  �  D  )  c  y  s  4  3    �  �  s      @  �  o  D  �  �  �  �  	  s  d  �  �  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  E�  a  ]  Z  W  S  P  M  G  ?  8  1  )  "           �   �   �  w  �  �  �  �  �  �    o  \  F  /    �  �  �  4  �  1   �  �  �  �  �  �  y  c  O  :  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  S  9    �  �  *  G  D  B  =  7  )      �  �  �  �  �  �  �  Y  .    �  �  �  �     N  r  �  �  �  �  o  X  1    �  �  J  �  R  u    �  �  �  �  �  �  �  c  (  �  �  I  �  �  >  �  9  �  L  -  �  �  �  �  �  �  �  �  �  �  �  \  0  �  �  W  �  v  �  �  �  �  �  �  h  L  /    �  �  �  Y  '  �  �  }  8  �  �  #  &      �  �  �  �  �  �  �  z  k  W  C  %  �  �  �  c  ,      �  �  �  �  �  �  Q  9    �  �  �  �  T  	  �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  c  P  =  +    	S  	�  	�  	�  	�  	�  	U  	)  �  �    �  �    �  Y  �  $  �  �  �  w  f  S  @  5  B  5    �  �  �  �  b  1  �  �  r  �  t  �  �  �  o  V  C  9  @  J  K  E  8  $    �  �  �  S  �  �        
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  J    �  �  k    �  M  �  �  �  �  �  �  �  �  y  X  0    �  �  7  �  d  �  |  �  �  �  �  �  �  �  �  �  �  �  |  t  h  Z  =    �  �  G   �  �  �  �  �  h  C    �  �  �  G    �  l  *  �  �  >  �  �  �  �  �  #  }  �  2  �  4  u  �  �  G  �  P  �      
�  �  7  .  &            �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  \  8    �  �  �  9  �  �    1  %        �  �  �  �  �  �  �  �  w  `  J  2     �   �    �  �  �  �  �  �  �  v  _  H  1       �  �  �  �  h  F  �  �  �  �  �  w  T  '  �  �  �  �  �  �    N  �  3  �  i  �  �  �  �  �  �  �  �    |  u  l  c  Y  P  F  <  1  '      �  �  �  �  j  D    �  �  h  )  �  �  F  �  �  I  �  �  2  =  9  /  U  8        �  �  w  !  �    w  �    Y  t  �  �  �              �  �  �  A  �  �  X     �  H   �  �  �  �  �          �  �  �  �  o  <  �  r  �  c  �  S  �  �  �  �  �  h  Q  D  ;  .      �  �  �  �  �  e  '  �  j  Y  H  8  %    �  �  �  �  �  �  �  �  p  Y  >     �   �  �  �  �  �  �  q  K    �  �  5  �  [  �  �  	  W  �  �  J  �  �  �  �  �  �  �  �  �  }  g  O  /    �  �  �  �  �  �  �  �  �  �  �  �  �  p  _  M  ;  )      �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \    �  p  >    W  k  �  �    6  Q  d  e  Y  C  "  �  �  <  �  �    .  !  �  �  �  �  �  �  �  �  �  ~  k  U  @  /      �  �  �  V  ~  �  	  	  	  	  	  �  �  �  l  "  �  V    �  �  1  U  �  �  q  b  Y  T  Q  K  @  1      �  �  �  d  ;    �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  N  5      �  �  �  �  �  �  }  `  A  !    �  �  �  �  �  �  �  z  `  H  4  (    �  �  �  �  �  �  t  e  O  7    �  �  �  p  7  �  �  1  �  
  �  �  �  �  �  �  �  �  �  �  e  F  $  �  �  �  ]    �  �  �  �  �  �  �  �  y  ]  ?  #  
  �  �  �  X  #  �  �  �  o  z  �  �  �  �  �  �  �  �  y  a  >    �  �  C  �  a  �  �  �  �  �  �  �  �  }  r  g  ^  V  M  E  =  4  *         �  1  V  i  f  W  E  .    �  �  �  U    �  r  �  �  :        
       �  �  �    
           �  �  �  �  �  �  �  �  �  �  �  �    e  E  !  �  �  �  `    �  �  4  �  '  m  G    �  �  �  y  O  )    �  �  �  t  B    �  �  �  I  _  u  �  �  �  }  c  C    �  �  �  P    �  �  7  �  �  /  �  �  �  �  �  �  �         �  �  �  �  �  [  "  �  �  c  �  	  	1  	7  	9  	@  	E  	(  �  �  �  E  �  �  0  �  �  �  �  U    �  �  �  �  {  \  ;    �  �  �  z  O  "  �  �  z  �  w  �  �  �  �  �  �  z  h  W  B  ,    �  �  �  �  �  �  �  �  l  �  �  �  �  �  �  �  �  w  H    �  |  )  �  �  V  I  \    2  5  3  &    �  �  �  [    �  �  �  H  �  �  F  �  �  �  u  %  �  q    �  4  
�  
  	�  �      �  �  2  V  p  �  	�  	�  	`  	0  �  �  �  `    �  �  g  /    �  0  �    j  �  �  q  Q  .  �  �  �  u  J    �  !  I  4    �  ^    �  �  �  �  �  �  �  �  �  r  _  I  4      �  �  �  �  �  �  �            �  �  �  �  �  �  �  �  �  �  �  �    g  �  	  	  �  �  �  �  �  a  1  �  �  �  L    �  �  S    l    i  c  U  >    �  �  �  �    `  D  )    �  �  �  �  �  }